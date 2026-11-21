"""Canais basais (ASE, ICESat-2), passo 2: seções transversais de canais por trilha repetida.

Ideia central: a profundidade da depressão superficial de um canal, medida contra as quilhas vizinhas
NO MESMO PERFIL, é uma diferença de alturas a poucos km de distância e no mesmo instante. Maré
oceânica, DAC/barômetro inverso, maré de equilíbrio e carga são praticamente iguais nos dois pontos
e se cancelam; o erro do geoide de comprimento de onda > 10 km também. Isso dispensa o modelo de
maré, que é a maior fonte de erro da altimetria sobre plataformas.

Para cada (plataforma, rgt, feixe):
1. x_atc é a coordenada ao longo da trilha de referência, comum a todos os ciclos; bins de 100 m.
   Por ciclo, mediana de h por bin (h = h_li, sem correção: as correções se cancelam na diferença).
2. Perfil empilhado: em cada ciclo remove-se a tendência local (mediana móvel de 10 km) e toma-se a
   mediana entre ciclos. Canais = mínimos com proeminência >= PROM m e largura (meia proeminência)
   de LMIN a LMAX (find_peaks sobre −perfil).
3. Por ciclo e por canal: centro e fundo por ajuste quadrático nos bins a ±w/2 do mínimo do ciclo
   (procurado a ±w do centro empilhado, para acompanhar a migração lateral); quilha = mediana dos
   bins em w..2w de cada lado (os dois lados exigidos); D = quilha − fundo (m de bordo livre).
   Exige >= 70% dos bins da janela ±2w preenchidos.
Saída: data/interim/canais/secoes.parquet (uma linha por canal × ciclo) e canais_empilhados.parquet
Parâmetros (PROM, LMIN, LMAX, janelas) são escolhas a testar por sensibilidade, não ótimos.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
D_IN = ROOT / "data" / "interim" / "canais" / "flutuante_ase.parquet"
OUT = ROOT / "data" / "interim" / "canais"
SINT = float(os.environ.get("SINT", "0"))      # teste sintético (m/a de bordo livre); 0 = dado real
SUF = os.environ.get("SUF", "")
BIN = 100.0
JAN_TEND = 10_000.0
PROM = 1.5          # m de bordo livre (~14 m de espessura hidrostática)
LMIN, LMAX = 300.0, 5000.0
MIN_CICLOS = 8
PLAT = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove", 6: "Getz"}


def grade_ciclos(g: pd.DataFrame):
    b = np.floor(g.x_atc.to_numpy() / BIN).astype(np.int64)
    g = g.assign(b=b)
    tab = g.groupby(["cycle", "b"]).h_elv.median().unstack("cycle")
    tab = tab.reindex(np.arange(tab.index.min(), tab.index.max() + 1))
    tempo = g.groupby("cycle").t_year.median()
    xy = g.groupby("b")[["x", "y"]].median().reindex(tab.index)
    return tab, tempo, xy


def destend(col: pd.Series) -> pd.Series:
    n = int(JAN_TEND / BIN)
    return col - col.rolling(n, center=True, min_periods=n // 3).median()


def mede(perfil: np.ndarray, c0: int, w: int):
    """perfil indexado por bin; c0 = bin do centro empilhado; w = meia-largura em bins."""
    n = len(perfil)
    a, z = max(c0 - 2 * w, 0), min(c0 + 2 * w + 1, n)
    jan = perfil[a:z]
    if np.isfinite(jan).mean() < 0.7:
        return None
    lo, hi = max(c0 - w, 0), min(c0 + w + 1, n)
    seg = perfil[lo:hi]
    if not np.isfinite(seg).any():
        return None
    cm = lo + int(np.nanargmin(seg))
    hw = max(w // 2, 2)
    ii = np.arange(max(cm - hw, 0), min(cm + hw + 1, n))
    ii = ii[np.isfinite(perfil[ii])]
    if len(ii) < 5:
        return None
    p = np.polyfit(ii - cm, perfil[ii], 2)
    if p[0] <= 0:
        fundo, centro = np.nanmin(perfil[ii]), float(cm)
    else:
        xv = -p[1] / (2 * p[0])
        if abs(xv) > hw:
            fundo, centro = np.nanmin(perfil[ii]), float(cm)
        else:
            fundo, centro = np.polyval(p, xv), cm + xv
    e = perfil[max(cm - 2 * w, 0):max(cm - w, 0)]
    d = perfil[min(cm + w + 1, n):min(cm + 2 * w + 1, n)]
    if np.isfinite(e).sum() < 3 or np.isfinite(d).sum() < 3:
        return None
    qe, qd = np.nanmedian(e), np.nanmedian(d)
    return dict(D=0.5 * (qe + qd) - fundo, quilha_e=qe - fundo, quilha_d=qd - fundo, centro_bin=centro)


def main():
    df = pd.read_parquet(D_IN)
    secoes, emp = [], []
    for (pl, rgt, beam), g in df.groupby(["plataforma", "rgt", "beam"]):
        if g.cycle.nunique() < MIN_CICLOS:
            continue
        tab, tempo, xy = grade_ciclos(g)
        if len(tab) < 2 * JAN_TEND / BIN:
            continue
        res = tab.apply(destend)
        pilha = res.median(axis=1, skipna=True)
        nval = res.notna().sum(axis=1)
        pilha[nval < MIN_CICLOS] = np.nan
        if pilha.notna().sum() < 2 * JAN_TEND / BIN:
            continue
        # lacunas interpoladas só para a detecção; picos com janela mal coberta são descartados
        prof = (-pilha).interpolate(limit_area="inside").to_numpy()
        valido = np.isfinite(prof)
        prof = np.where(valido, prof, np.nanmin(prof))
        pk, pr = find_peaks(prof, prominence=PROM, width=(LMIN / BIN, LMAX / BIN))
        for j, c0 in enumerate(pk):
            w = max(int(round(pr["widths"][j] / 2)), 3)
            a, z = max(c0 - 2 * w, 0), min(c0 + 2 * w + 1, len(pilha))
            if pilha.iloc[a:z].notna().mean() < 0.8 or a == 0 or z == len(pilha):
                continue
            cid = f"{PLAT[pl]}_{rgt}_{beam}_{int(tab.index[c0])}"
            emp.append(dict(canal=cid, plataforma=PLAT[pl], rgt=rgt, beam=beam, b0=int(tab.index[c0]),
                            x=xy.x.iloc[c0], y=xy.y.iloc[c0], prom=pr["prominences"][j], largura_m=pr["widths"][j] * BIN,
                            n_ciclos=int(nval.iloc[c0])))
            for cyc in tab.columns:
                perfil = tab[cyc].to_numpy()
                if SINT != 0.0:
                    # teste sintético: aprofundamento uniforme de SINT m/a de bordo livre no canal,
                    # forma de cosseno com meia-largura w (zero nas quilhas, w..2w), aplicado
                    # depois da detecção para manter os mesmos canais
                    q = np.arange(len(perfil)) - c0
                    forma = np.where(np.abs(q) < w, 0.5 * (1 + np.cos(np.pi * q / w)), 0.0)
                    perfil = perfil - SINT * (float(tempo[cyc]) - 2022.0) * forma
                r = mede(perfil, c0, w)
                if r is None:
                    continue
                cb = int(np.clip(round(r["centro_bin"]), 0, len(tab) - 1))
                secoes.append(dict(canal=cid, plataforma=PLAT[pl], rgt=rgt, beam=beam, cycle=int(cyc),
                                   t=float(tempo[cyc]), x=xy.x.iloc[cb], y=xy.y.iloc[cb],
                                   desloc_m=(r["centro_bin"] - c0) * BIN, **{k: r[k] for k in ("D", "quilha_e", "quilha_d")}))
    e = pd.DataFrame(emp)
    s = pd.DataFrame(secoes)
    if SINT == 0.0:
        e.to_parquet(OUT / "canais_empilhados.parquet", index=False)
    s.to_parquet(OUT / f"secoes{SUF}.parquet", index=False)
    print(e.groupby("plataforma").agg(n=("canal", "size"), prom=("prom", "median"), larg=("largura_m", "median")))
    print(s.groupby("plataforma").agg(n=("D", "size"), canais=("canal", "nunique")))


if __name__ == "__main__":
    main()
