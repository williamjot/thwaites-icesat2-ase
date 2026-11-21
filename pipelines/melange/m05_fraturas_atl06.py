"""Mélange e fraturas da Thwaites, passo 5: fraturas grandes no ATL06 (TEIS e língua oeste).

Entrada: data/interim/melange/thwaites_atl06.parquet (passo 1) e limiares.json (passo 2).
Só segmentos com hfw ≥ LIM_GROSSO dentro dos domínios TEIS e TWGT (produto basal v2, 2 km).
Por perfil (grânulo × feixe), em ordem de x_atc:
  resíduo r = hfw − mediana móvel de 1 km (51 segmentos de 20 m), em segmentos grossos;
  ruído σ_r = 1,4826·MAD de r no interior da TEIS (referência de gelo coeso), uma vez para tudo;
  fratura = sequência contínua de segmentos com r < −K·σ_r (K = 4) e profundidade mínima ≥ 3 m,
            OU lacuna de ≥ 3 segmentos (≥ 60 m) entre segmentos grossos, sem nuvem (nuvem → descarta).
  Largura ATL06: o segmento tem 40 m com passo de 20 m, então só fraturas ≳ 60–100 m aparecem;
  fendas menores ficam para o ATL03.
Densidade = fraturas por km de perfil grosso, por célula de 2 km e por época.
Zonas: margem de cisalhamento = TEIS a ≤ 5 km da TWGT; interior da TEIS = TEIS a > 10 km; TWGT.
Saídas: outputs/melange/fraturas_eventos.parquet, fraturas_serie.csv, fraturas_ano_2km.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "interim" / "melange"
OUT = ROOT / "outputs" / "melange"
K = 4.0
PROF_MIN = 3.0
RNG = np.random.default_rng(20261118)


def k2(x, y):
    return (np.round(x / 2000) * 2).astype(int).astype(str) + "_" + (np.round(y / 2000) * 2).astype(int).astype(str)


def main():
    lim = json.loads((DIR / "limiares.json").read_text())
    nv = pd.read_csv(DIR / "nivel_agua_campanha.csv")
    dom = pd.read_csv(ROOT / "manuscrito" / "figuras" / "dados" / "R_fig1_dominio_basal_2km.csv")
    dom = dom[dom.unidade.str.startswith("Thwaites")].copy()
    dom["k"] = k2(dom.x_km.to_numpy() * 1000, dom.y_km.to_numpy() * 1000)
    tw = dom[dom.unidade == "Thwaites_TWGT"][["x_km", "y_km"]].to_numpy() * 1000
    te = dom[dom.unidade == "Thwaites_TEIS"].copy()
    te["d_twgt"] = cKDTree(tw).query(te[["x_km", "y_km"]].to_numpy() * 1000)[0] / 1000
    zona = dict(zip(te.k, np.where(te.d_twgt <= 5, "margem", np.where(te.d_twgt > 10, "interior", "TEIS meio"))))
    zona.update({k: "TWGT" for k in dom[dom.unidade == "Thwaites_TWGT"].k})

    d = pd.read_parquet(DIR / "thwaites_atl06.parquet",
                        columns=["x", "y", "t", "hf", "xatc", "gran", "beam", "rgt", "est", "nuvem", "seg"])
    d["k"] = k2(d.x.to_numpy(), d.y.to_numpy())
    d["zona"] = d.k.map(zona)
    d = d[d.zona.notna()].copy()
    ano = np.floor(d.t).astype(int)
    d["ano"] = np.where((d.est == 1) & (d.t - ano > 0.8), ano + 1, ano)
    d = d.merge(nv, on=["ano", "est"], how="left")
    d["hfw"] = d.hf - d.nivel.fillna(nv.nivel.median())
    d = d[d.hfw >= lim["LIM_GROSSO"]].sort_values(["gran", "beam", "xatc"]).reset_index(drop=True)
    d["med"] = (d.groupby(["gran", "beam"]).hfw
                .transform(lambda s: s.rolling(51, center=True, min_periods=15).median()))
    d["r"] = d.hfw - d.med
    sig = 1.4826 * np.nanmedian(np.abs(d.loc[d.zona == "interior", "r"] - np.nanmedian(d.loc[d.zona == "interior", "r"])))
    print(f"σ do resíduo no interior da TEIS: {sig:.2f} m -> limiar {K * sig:.2f} m (mín. {PROF_MIN} m)")
    lim_r = -max(K * sig, PROF_MIN)

    eventos = []
    comp = []          # comprimento de perfil grosso por (k, ano, est, rgt)
    for (gr, bm), p in d.groupby(["gran", "beam"], sort=False):
        xa = p.xatc.to_numpy(); r = p.r.to_numpy(); nu = p.nuvem.to_numpy()
        kk = p.k.to_numpy()
        a0, e0, rg = int(p.ano.iloc[0]), int(p.est.iloc[0]), int(p.rgt.iloc[0])
        comp.append(pd.DataFrame({"k": kk, "km": 0.02}).groupby("k").km.sum().rename("km").reset_index()
                    .assign(ano=a0, est=e0, rgt=rg))
        # depressões
        low = r < lim_r
        if low.any():
            idx = np.flatnonzero(low)
            grupos = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for gi in grupos:
                c = gi[np.argmin(r[gi])]
                eventos.append(("depressão", p.x.iat[c], p.y.iat[c], kk[c], a0, e0, rg, -r[c], (len(gi) + 1) * 20.0))
        # lacunas entre segmentos grossos (≥ 60 m), sem nuvem nos vizinhos
        gap = np.diff(xa)
        for i in np.flatnonzero((gap >= 60) & (gap <= 1000)):
            if nu[i] > 0 or nu[i + 1] > 0:
                continue
            eventos.append(("lacuna", 0.5 * (p.x.iat[i] + p.x.iat[i + 1]), 0.5 * (p.y.iat[i] + p.y.iat[i + 1]), kk[i],
                            a0, e0, rg, np.nan, gap[i]))
    ev = pd.DataFrame(eventos, columns=["tipo", "x", "y", "k", "ano", "est", "rgt", "prof", "largura"])
    ev["zona"] = ev.k.map(zona)
    ev.to_parquet(OUT / "fraturas_eventos.parquet", index=False)
    cp = pd.concat(comp).groupby(["k", "ano", "est", "rgt"]).km.sum().reset_index()
    cp["zona"] = cp.k.map(zona)

    # densidade por zona e ano (todas as estações), bootstrap por RGT
    lin = []
    for (zn, a), c in cp.groupby(["zona", "ano"]):
        e = ev[(ev.zona == zn) & (ev.ano == a)]
        for tipo in ["depressão", "lacuna", "ambos"]:
            ee = e if tipo == "ambos" else e[e.tipo == tipo]
            nr = ee.groupby("rgt").size().reindex(c.rgt.unique(), fill_value=0)
            kr = c.groupby("rgt").km.sum()
            u = kr.index.to_numpy()
            bt = [nr.loc[s].sum() / kr.loc[s].sum() for s in (RNG.choice(u, len(u)) for _ in range(300))]
            lin.append(dict(zona=zn, ano=a, tipo=tipo, km=kr.sum(), n=int(nr.sum()), dens=nr.sum() / kr.sum(),
                            lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5), n_rgt=len(u)))
    s = pd.DataFrame(lin)
    s.to_csv(OUT / "fraturas_serie.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(s[s.tipo != "lacuna"].round(3).to_string())
    g = ev.assign(gx=(np.round(ev.x / 2000) * 2).astype(int), gy=(np.round(ev.y / 2000) * 2).astype(int))
    g.groupby(["ano", "gx", "gy", "tipo"]).size().rename("n").reset_index().to_csv(OUT / "fraturas_ano_2km.csv",
                                                                                   index=False)


if __name__ == "__main__":
    main()
