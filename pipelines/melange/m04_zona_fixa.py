"""Mélange e fraturas da Thwaites, passo 4: gelo grosso em zonas fixas, por época, e teste de mudança.

Entrada: outputs/melange/classes_janelas.parquet (passo 3).
Zonas (fixas no tempo, para a amostragem orbital não criar tendência):
  Z_melange  células de 2 km fora da TEIS/TWGT em que, em 2019–2021 (linha de base), ≥ 30% das janelas
             eram gelo grosso (mélange ou coeso), com ≥ 5 janelas. Define-se só com a linha de base.
  Z_evento   os 215 blocos de 2 km do evento de 2024 (evento_2024/outputs/fase7c_degrau), janela a ≤ 1 km
             do centro.
  TEIS, TWGT domínios do produto basal v2.
Por época (estação × ano): fração de janelas com gelo grosso (mélange + coeso) e fração coesa;
IC 95% por bootstrap das passagens (RGT), 300 réplicas.
Mudança: média 2019–2022 contra 2024–2025 (2023 fica de fora como transição), com IC pela diferença
das réplicas emparelhadas por época; e a mesma comparação por estação (o sinal tem de repetir nas
quatro). Saída: outputs/melange/serie_zonas.csv, mudanca_zonas.csv, zona_melange_2km.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "melange"
RNG = np.random.default_rng(20261118)
NB = 300


def main():
    j = pd.read_parquet(OUT / "classes_janelas.parquet")
    j = j[j.classe != "sem"].copy()
    j["grosso"] = j.classe.isin(["mélange", "coeso"])
    j["coeso"] = j.classe == "coeso"
    j["gx"] = (np.round(j.x / 2000) * 2).astype(int)
    j["gy"] = (np.round(j.y / 2000) * 2).astype(int)
    j["epoca"] = j.ano + np.select([j.est == 0, j.est == 1, j.est == 2, j.est == 3], [0.54, 0.04, 0.79, 0.29])

    base = j[(j.ano <= 2021) & (j.regiao == "baía")].groupby(["gx", "gy"]).agg(n=("grosso", "size"),
                                                                               f=("grosso", "mean"))
    zm = base[(base.n >= 5) & (base.f >= 0.3)].reset_index()
    zm.to_csv(OUT / "zona_melange_2km.csv", index=False)
    j = j.merge(zm[["gx", "gy"]].assign(z_mel=True), on=["gx", "gy"], how="left")
    j["z_mel"] = j.z_mel.fillna(False).astype(bool)

    ev = pd.read_csv(ROOT / "evento_2024" / "outputs" / "fase7c_degrau" / "blocos_antes_depois.csv")
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(ev[["x", "y"]].to_numpy()).query(j[["x", "y"]].to_numpy())
    j["z_ev"] = dist <= 1000.0

    zonas = {"zona de mélange (2019–21)": j.z_mel, "blocos do evento de 2024": j.z_ev,
             "TEIS": j.regiao == "TEIS", "TWGT": j.regiao == "TWGT"}
    print({k: int(v.sum()) for k, v in zonas.items()}, "| células na zona de mélange:", len(zm))

    ser, mud = [], []
    for nome, msk in zonas.items():
        g = j[msk]
        reps_ep = {}
        for ep, ge in g.groupby("epoca"):
            if ge.rgt.nunique() < 3 or len(ge) < 20:
                continue
            rg = ge.rgt.to_numpy(); u = np.unique(rg)
            gro = ge.grosso.to_numpy(); coe = ge.coeso.to_numpy()
            idx = {r: np.where(rg == r)[0] for r in u}
            bt = np.empty((NB, 2))
            for b in range(NB):
                ii = np.concatenate([idx[r] for r in RNG.choice(u, len(u))])
                bt[b] = gro[ii].mean(), coe[ii].mean()
            reps_ep[ep] = bt
            ser.append(dict(zona=nome, epoca=ep, est=int(ge.est.iloc[0]), n=len(ge), n_rgt=len(u),
                            grosso=gro.mean(), grosso_lo=np.percentile(bt[:, 0], 2.5),
                            grosso_hi=np.percentile(bt[:, 0], 97.5), coeso=coe.mean(),
                            coeso_lo=np.percentile(bt[:, 1], 2.5), coeso_hi=np.percentile(bt[:, 1], 97.5)))
        s = pd.DataFrame([r for r in ser if r["zona"] == nome])
        for est_sel, rot in [(None, "todas")] + [(e, n) for e, n in enumerate(["JJA", "DJF", "SON", "MAM"])]:
            ss = s if est_sel is None else s[s.est == est_sel]
            a = ss[ss.epoca < 2023].epoca.tolist(); d = ss[ss.epoca >= 2024].epoca.tolist()
            if len(a) < 2 or len(d) < 1:
                continue
            for k, var in enumerate(["grosso", "coeso"]):
                dif = (np.mean([reps_ep[e][:, k] for e in d], axis=0) - np.mean([reps_ep[e][:, k] for e in a], axis=0))
                obs = ss[ss.epoca >= 2024][var].mean() - ss[ss.epoca < 2023][var].mean()
                mud.append(dict(zona=nome, estacao=rot, variavel=var, antes=ss[ss.epoca < 2023][var].mean(),
                                depois=ss[ss.epoca >= 2024][var].mean(), dif=obs,
                                lo=np.percentile(dif, 2.5), hi=np.percentile(dif, 97.5),
                                n_ep_antes=len(a), n_ep_depois=len(d)))
    pd.DataFrame(ser).to_csv(OUT / "serie_zonas.csv", index=False)
    m = pd.DataFrame(mud)
    m.to_csv(OUT / "mudanca_zonas.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(m.round(3).to_string())


if __name__ == "__main__":
    main()
