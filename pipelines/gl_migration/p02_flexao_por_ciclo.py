"""Migração da linha de aterramento — passo 2: posição da flexão de maré em cada passagem.

Por perfil (rgt, par) do passo 1:
1. Intervalos de 100 m ao longo da trilha; mediana de r = h − REMA por ciclo e intervalo.
2. Por intervalo, ajuste robusto (Huber) r = a + b·(t − t̄) + β·(T − T̄) sobre os ciclos: remove a
   superfície média e a tendência (afinamento) sem remover a maré. Desvio do ciclo k:
   d_k(x) = r_k(x) − a(x) − b(x)·(t_k − t̄)  ≈ β(x)·(T_k − T̄) + ruído.
3. Amplitude do ciclo no gelo livre (6–12 km a jusante da linha do BedMachine): A_k = mediana de d_k.
   Só ciclos com |T_k − T̄| ≥ 0,30 m e sinal de A_k igual ao de T_k − T̄ entram.
4. Perfil normalizado n_k = (d_k − c_k)/A_k, com c_k = mediana de d_k no gelo apoiado (3–10 km a
   montante), suavizado por mediana móvel de 500 m. Posição de meia flexão x50_k: o primeiro x,
   vindo do apoiado, a partir do qual n_k ≥ 0,5 por 1 km contínuo. Também x20_k (n_k ≥ 0,2), mais
   perto do limite de flexão.
Coordenada x: distância ao longo da trilha a partir do cruzamento com a linha do BedMachine
(positivo = a jusante). Inclui x, y da posição (para comparar com InSAR e IS2GZANT).

Limitações declaradas: (i) o termo a + b·t não acompanha ondulações que avançam com o gelo, o que
aumenta o ruído nas plataformas rápidas; (ii) com a linha migrando com a maré, d_k não é exatamente
β·ΔT de uma linha fixa: x50_k é a posição efetiva de meia flexão no ciclo k; (iii) a linha do
BedMachine só serve de origem; os resultados são diferenças de posição.

Saída: outputs/gl_migration/flexao_por_ciclo_<plataforma>.parquet e perfis_qc_<plataforma>.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas as _pd


def median_filter(a, size=5, mode=None):
    """Mediana móvel centrada que ignora NaN (o median_filter do scipy corrompe vizinhos de NaN)."""
    return _pd.Series(a, dtype=float).rolling(size, center=True, min_periods=(size + 1) // 2).median().to_numpy().copy()

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "interim" / "gl_migration"
OUT = ROOT / "outputs" / "gl_migration"
B = 100.0
DT_MIN = 0.30
JANELA = 5            # intervalos (500 m) na mediana móvel
CONTIGUO = 10         # intervalos (1 km) contínuos acima do limiar


def huber_wls(X, y, k=1.345, it=20):
    w = np.ones(len(y))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = np.abs(r) / (k * s)
        w = np.where(u <= 1, 1.0, 1.0 / u)
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return beta


def primeiro_cruzamento(xb, n, lim):
    acima = np.nan_to_num(n, nan=-9) >= lim
    run = 0
    for i in range(len(acima)):
        run = run + 1 if acima[i] else 0
        if run >= CONTIGUO:
            return xb[i - CONTIGUO + 1]
    return np.nan


def cruzamentos(g: pd.DataFrame):
    """Cruzamentos da linha do BedMachine ao longo do perfil. Uma trilha pode cruzar a linha mais de
    uma vez (ilhas, reentrâncias); cada cruzamento vira um perfil próprio, orientado do apoiado para
    o flutuante e limitado pelos cruzamentos vizinhos."""
    sm = g.groupby(np.floor(g.s / B) * B).dist_gl.median()
    s, dgl = sm.index.values, sm.values
    sinal = np.sign(dgl)
    idx = np.where((sinal[:-1] * sinal[1:]) < 0)[0]
    cz = []
    for i in idx:
        sc = s[i] + (s[i + 1] - s[i]) * (0 - dgl[i]) / (dgl[i + 1] - dgl[i])
        cz.append((sc, 1 if dgl[i] < 0 else -1))
    # descarta cruzamentos a menos de 3 km de outro (linha recortada; ambíguo)
    bons = []
    for k, (sc, o) in enumerate(cz):
        viz = [abs(sc - c2) for j, (c2, _) in enumerate(cz) if j != k]
        if not viz or min(viz) > 3000:
            lim_lo = max([c2 for c2, _ in cz if c2 < sc], default=-np.inf)
            lim_hi = min([c2 for c2, _ in cz if c2 > sc], default=np.inf)
            bons.append((sc, o, lim_lo, lim_hi))
    return bons


def processa_perfil(g: pd.DataFrame, sg: float, orient: int, lim_lo: float, lim_hi: float):
    g = g[(g.s > lim_lo) & (g.s < lim_hi)]
    g = g.assign(xb=np.floor(orient * (g.s - sg) / B) * B + B / 2)
    g = g[(g.xb > -12000) & (g.xb < 14000)]
    if g.xb.min() > -4000 or g.xb.max() < 8000:
        return None
    pb = g.groupby(["ciclo", "xb"]).agg(r=("r", "median"), T=("T", "median"), t=("t", "median"),
                                        X=("x", "median"), Y=("y", "median")).reset_index()
    cic = pb.groupby("ciclo").agg(T=("T", "median"), t=("t", "median"))
    if len(cic) < 8:
        return None
    grade = np.arange(-12000 + B / 2, 14000, B)
    piv = pb.pivot(index="xb", columns="ciclo", values="r").reindex(index=grade, columns=cic.index)
    xb = piv.index.values
    Tm, tm = cic["T"].mean(), cic["t"].mean()
    dT = (cic["T"] - Tm).values
    dt = (cic["t"] - tm).values
    D = np.full(piv.shape, np.nan)
    beta = np.full(len(xb), np.nan)
    for i, row in enumerate(piv.values):
        ok = np.isfinite(row)
        if ok.sum() < 8:
            continue
        X = np.c_[np.ones(ok.sum()), dt[ok], dT[ok]]
        a, b, be = huber_wls(X, row[ok])
        beta[i] = be
        D[i] = row - a - b * dt
    gmask = (xb > -10000) & (xb < -3000)
    fmask = (xb > 6000) & (xb < 12000)
    # posição (x, y) de cada intervalo para georreferenciar x50
    pos = pb.groupby("xb")[["X", "Y"]].median().reindex(xb).interpolate(limit_direction="both")
    far = np.nanmedian(beta[fmask])
    # gabarito: β médio normalizado do próprio perfil (todas as passagens), suavizado
    tmpl = median_filter(np.nan_to_num(beta / far, nan=np.nan), size=JANELA, mode="nearest")
    tmpl = pd.Series(tmpl).interpolate(limit=5).to_numpy()
    tx50 = primeiro_cruzamento(xb, tmpl, 0.5)
    janela = (xb > -6000) & (xb < 10000)
    desloc = np.arange(-50, 51)            # deslocamentos de −5 a +5 km, passo de 100 m
    linhas = []
    for j, cyc in enumerate(cic.index):
        d = D[:, j]
        c = np.nanmedian(d[gmask])
        A = np.nanmedian(d[fmask]) - c
        nf = np.isfinite(d[fmask]).sum()
        ng = np.isfinite(d[gmask]).sum()
        reg = dict(ciclo=cyc, t=cic.loc[cyc, "t"], T=cic.loc[cyc, "T"], dT=dT[j], A=A, nf=nf, ng=ng)
        if abs(dT[j]) >= DT_MIN and np.sign(A) == np.sign(dT[j]) and nf > 20 and ng > 20:
            n = (d - c) / A
            # ajuste de gabarito: deslocamento δ que minimiza a mediana de |n(x) − gabarito(x − δ)|
            custo = np.full(len(desloc), np.nan)
            for q, k in enumerate(desloc):
                tk = np.roll(tmpl, k)
                if k > 0:
                    tk[:k] = tmpl[0]
                elif k < 0:
                    tk[k:] = tmpl[-1]
                ok = janela & np.isfinite(n) & np.isfinite(tk)
                if ok.sum() >= 40:
                    custo[q] = np.mean(np.abs(n[ok] - tk[ok]))
            if np.isfinite(custo).sum() > 20:
                q = int(np.nanargmin(custo))
                reg["delta"] = desloc[q] * B
                reg["custo"] = custo[q]
                # nitidez: custo médio a ±1 km do mínimo menos o mínimo (curva rasa = mal determinado)
                viz = custo[max(0, q - 10):q + 11]
                reg["nitidez"] = np.nanmax(viz) - custo[q]
                reg["borda"] = q in (0, len(desloc) - 1)
                if np.isfinite(tx50):
                    xc = tx50 + reg["delta"]
                    reg["x50"] = xc
                    k = int(np.clip(np.searchsorted(xb, xc), 0, len(xb) - 1))
                    reg["x50_X"], reg["x50_Y"] = pos.iloc[k].values
        linhas.append(reg)
    out = pd.DataFrame(linhas)
    qc = dict(beta_far=far, n_ciclos=len(cic), Tm=Tm, tm=tm, tx50=tx50,
              beta_perfil=beta.tolist(), xb=xb.tolist())
    return out, qc


def main(plataformas):
    OUT.mkdir(parents=True, exist_ok=True)
    for p in plataformas:
        d = pd.read_parquet(IN / f"perfis_{p}.parquet")
        res, qcs = [], []
        for (rgt, par), g in d.groupby(["rgt", "par"]):
            for k, (sg, orient, lo, hi) in enumerate(cruzamentos(g)):
                r = processa_perfil(g, sg, orient, lo, hi)
                if r is None:
                    continue
                o, qc = r
                o["rgt"], o["par"], o["cz"] = rgt, par, k
                res.append(o)
                qcs.append(dict(rgt=rgt, par=par, cz=k, orient=orient, **qc))
        res = pd.concat(res, ignore_index=True)
        qcs = pd.DataFrame(qcs)
        res.to_parquet(OUT / f"flexao_por_ciclo_{p}.parquet", index=False)
        qcs.to_parquet(OUT / f"perfis_qc_{p}.parquet", index=False)
        print(p, "perfis", len(qcs), "beta_far mediano", round(np.nanmedian(qcs.beta_far), 2),
              "ciclos com x50", int(res["x50"].notna().sum()) if "x50" in res else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or ["Cosgrove"])
