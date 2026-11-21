"""Mélange e fraturas da Thwaites, passo 10: a rugosidade do ATL06 que sobe na margem da TEIS é dano novo
ou gelo mais rugoso transportado para dentro da faixa fixa (euleriana)?

Entrada: outputs/melange/classes_janelas.parquet (janelas de 1 km; rug = desvio robusto do bordo livre
grosso sem tendência). Domínio TEIS do produto basal v2; margem = TEIS a ≤ 5 km da TWGT; interior > 10 km.
População: janelas com f_grosso ≥ 0,72 (critério de gelo grosso da classe coesa, SEM o corte de rug, para
não truncar o que se quer medir) e, à parte, só as coesas (como no número reportado antes).

1. Euleriano: média anual de rug na faixa fixa (reproduz o 2,3 → 3,5 m), IC por bootstrap de RGT.
2. Previsão por advecção pura: o centro de cada janela é retroadvectado até T_REF (ITS_LIVE anual,
   interpolado no tempo; mesmo integrador do passo 9). Campo material de referência = janelas de
   2019–20 (todo gelo grosso da caixa, não só a TEIS) em bins de 1,5 km no referencial do gelo (≥ 3 janelas). Para cada janela de 2024–25 na
   faixa euleriana com bin material visto em 2019–20: rug previsto = média do bin em 2019–20.
   Observado − previsto, IC por bootstrap emparelhado de RGT (as duas épocas reamostradas).
   Dano novo → observado > previsto no mesmo gelo; advecção → observado ≈ previsto.
Saídas: outputs/melange/rug_euleriano.csv, rug_previsao.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from m09_advecao_fendas import Campo, T_REF  # noqa: E402

OUT = ROOT / "outputs" / "melange"
BIN = 1500.0
NB = 500
RNG = np.random.default_rng(20261118)


def main():
    dom = pd.read_csv(ROOT / "manuscrito" / "figuras" / "dados" / "R_fig1_dominio_basal_2km.csv")
    tw = dom[dom.unidade == "Thwaites_TWGT"][["x_km", "y_km"]].to_numpy() * 1000
    te = dom[dom.unidade == "Thwaites_TEIS"][["x_km", "y_km"]].to_numpy() * 1000
    j = pd.read_parquet(OUT / "classes_janelas.parquet")
    j = j[j.rug.notna() & (j.f_grosso >= 0.72)].copy()      # referência: todo gelo grosso da caixa
    j["d_tw"] = cKDTree(tw).query(j[["x", "y"]].to_numpy())[0] / 1000
    j["faixa"] = np.where(j.regiao != "TEIS", "fora", np.where(j.d_tw <= 5, "margem",
                                                               np.where(j.d_tw > 10, "interior", "meio")))
    print(j.groupby(["faixa", "classe"]).size().unstack(fill_value=0))

    campo = Campo(j.x.to_numpy(), j.y.to_numpy())
    j["x0"], j["y0"] = campo.leva(j.x.to_numpy(), j.y.to_numpy(), j.t.to_numpy(), T_REF)
    print("sem velocidade (descartadas):", int((~np.isfinite(j.x0)).sum()), "de", len(j))
    j = j[np.isfinite(j.x0) & np.isfinite(j.y0)].copy()
    j["b"] = (np.floor(j.x0 / BIN).astype(int).astype(str) + "_" + np.floor(j.y0 / BIN).astype(int).astype(str))
    j["desl_km"] = np.hypot(j.x - j.x0, j.y - j.y0) / 1000

    eul, prev = [], []
    for pop, jj in [("f_grosso ≥ 0,72", j), ("só coesas", j[j.classe == "coeso"])]:
        for fx in ("margem", "interior"):
            g = jj[jj.faixa == fx]
            for a, ga in g.groupby("ano"):
                u = ga.rgt.unique(); idx = {r: np.flatnonzero(ga.rgt.to_numpy() == r) for r in u}
                v = ga.rug.to_numpy()
                bt = [v[np.concatenate([idx[r] for r in RNG.choice(u, len(u))])].mean() for _ in range(NB)]
                eul.append(dict(pop=pop, faixa=fx, ano=a, n=len(ga), n_rgt=len(u), rug=v.mean(),
                                lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5)))
            # previsão por advecção: referência = TODO o TEIS em 2019–20 (o gelo que chega à faixa
            # pode ter estado fora dela em 2019–20); alvo = faixa em 2024–25
            ref = jj[jj.ano <= 2020]
            alvo = g[g.ano >= 2024]
            rr, ra = ref.rgt.unique(), alvo.rgt.unique()
            ir = {r: np.flatnonzero(ref.rgt.to_numpy() == r) for r in rr}
            ia = {r: np.flatnonzero(alvo.rgt.to_numpy() == r) for r in ra}
            res = []
            for k in range(NB + 1):
                sr = np.arange(len(ref)) if k == 0 else np.concatenate([ir[r] for r in RNG.choice(rr, len(rr))])
                sa = np.arange(len(alvo)) if k == 0 else np.concatenate([ia[r] for r in RNG.choice(ra, len(ra))])
                rb = ref.iloc[sr].groupby("b").rug.agg(["mean", "size"])
                rb = rb[rb["size"] >= 3]["mean"]
                al = alvo.iloc[sa]
                ok = al.b.isin(rb.index).to_numpy()
                if ok.sum() < 10:
                    res.append((np.nan, np.nan, 0)); continue
                res.append((al.rug.to_numpy()[ok].mean(), rb.reindex(al.b[ok]).to_numpy().mean(), ok.mean()))
            res = np.array(res)
            d = res[1:, 0] - res[1:, 1]
            # referência euleriana ingênua: média da faixa em 2019–20
            ing = g[g.ano <= 2020].rug.mean()
            prev.append(dict(pop=pop, faixa=fx, n_alvo=len(alvo), frac_com_ref=res[0, 2], obs=res[0, 0],
                             previsto=res[0, 1], obs_menos_prev=res[0, 0] - res[0, 1],
                             lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                             faixa_2019_20=ing, desl_medio_km=alvo.desl_km.mean()))
    E = pd.DataFrame(eul); P = pd.DataFrame(prev)
    E.to_csv(OUT / "rug_euleriano.csv", index=False); P.to_csv(OUT / "rug_previsao.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(E.pivot_table(index="ano", columns=["pop", "faixa"], values="rug").round(2).to_string())
        print("\nPrevisão por advecção (2024–25, rug em m):")
        print(P.round(2).to_string())


if __name__ == "__main__":
    main()
