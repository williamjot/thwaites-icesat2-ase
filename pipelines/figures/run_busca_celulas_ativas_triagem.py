# -*- coding: utf-8 -*-
"""Triagem das células ativas (2026-11-19): roda, em cada célula extraída por run_busca_celulas_ativas_scan.py,
as MESMAS funções da série publicada (run_grounded_teis_method_series: load_observations, joint_window_fit,
trend_and_seasonality), sem firn, JJA 2019 … JJA 2025, só grânulos com pareamento de QC verificado.

Critério de "zigue-zague visível" fixado ANTES de olhar as séries:
  * ativa: tendência ≤ −1 m/ano com IC 95% abaixo de 0;
  * sazonal: JJA − DJF (resíduo da tendência) > 0 com IC 95% acima de 0;
  * visível: nº de transições verão → inverno com Δh ≥ 0 (a curva sobe) — é o que faz a figura parecer a do
    interior. Ordena por esse número e, no empate, pela razão (JJA − DJF) / |tendência·0,5 ano|.
Saída: outputs/grounded_H_timeseries/busca_celulas_ativas/triagem.csv (todas as células, inclusive as que falham).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipelines" / "figures"))
import run_grounded_teis_method_series as M                          # noqa: E402
from thwaites import load_config                                     # noqa: E402

RAW = ROOT / "data" / "processed" / "busca_celulas_ativas_raw.parquet"
CAND = ROOT / "data" / "processed" / "busca_celulas_ativas_candidatas.csv"
OUT = ROOT / "outputs" / "grounded_H_timeseries" / "busca_celulas_ativas"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(CAND)
    import os
    if os.environ.get("MAXC"):
        cand = cand.head(int(os.environ["MAXC"]))
    raw = pd.read_parquet(RAW)
    grupos = dict(tuple(raw.groupby("cell")))
    del raw
    cur = {}
    orig = pd.read_parquet

    def rp(path, *a, **k):                       # entrega só a célula da vez a load_observations
        return cur["df"] if str(path) == "__CELULA__" else orig(path, *a, **k)
    pd.read_parquet = rp
    M.RAW_PARQUET = "__CELULA__"
    M.TARGET_WINDOWS = M.build_windows(2025)
    M.WINDOW_ORDER = {w: i for i, w in enumerate(M.TARGET_WINDOWS)}
    M.COM_FIRN = False
    M.ACEITA_JOIN_NAO_VERIFICADO = False
    M.BOOTSTRAP_DRAWS = 5000
    cfg = load_config()
    rows = []
    for k, r in enumerate(cand.itertuples(index=False)):
        base = {"cell": r.cell, "geleira": r.geleira, "x": r.x, "y": r.y, "dhdt_mapa": r.dhdt_ice}
        if r.cell not in grupos:
            rows.append({**base, "erro": "sem pontos"}); continue
        cur["df"] = grupos[r.cell]
        M.CELL_NAME, M.CX, M.CY = r.cell, float(r.x), float(r.y)
        try:
            obs, audit = M.load_observations(cfg)
            ser = M.joint_window_fit(obs)
            if len(ser) < 13:
                rows.append({**base, "erro": f"{len(ser)} janelas", "n_janelas": len(ser)}); continue
            met = M.trend_and_seasonality(ser, seed=9561)
        except Exception as e:                                           # noqa: BLE001
            rows.append({**base, "erro": str(e)[:80]}); continue
        dj = np.asarray(met["djf_to_jja_deltas_m"]); jd = np.asarray(met["jja_to_djf_deltas_m"])
        rows.append({**base, "erro": "", "n_janelas": len(ser), "n_obs": int(audit["observacoes_finais"]),
                     "slope": met["slope_m_per_year"], "slope_sig": met["slope_sigma_m_per_year"],
                     "slope_hi": met["slope_ci95_high"], "saz": met["JJA_minus_DJF_detrended_m"],
                     "saz_lo": met["JJA_minus_DJF_ci95_low"], "saz_hi": met["JJA_minus_DJF_ci95_high"],
                     "n_vi_sobe": int((dj >= 0).sum()), "n_iv_desce": int((jd < 0).sum()),
                     "sigma_med": float(ser.sigma_h_ice.median()),
                     "repetivel": met["repeatable_seasonal_variability"]})
        ser.to_csv(OUT / f"serie_{r.cell}.csv", index=False)
        if k % 25 == 0:
            print(f"{k}/{len(cand)}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "triagem.csv", index=False)
    t = pd.DataFrame(rows)
    ok = t.erro.eq("")
    t["ativa"] = ok & (t.slope <= -1.0) & (t.slope_hi < 0)
    t["sazonal"] = ok & (t.saz > 0) & (t.saz_lo > 0)
    t["razao"] = t.saz / (t.slope.abs() * 0.5)
    t = t.sort_values(["ativa", "sazonal", "n_vi_sobe", "razao"], ascending=False)
    t.to_csv(OUT / "triagem.csv", index=False)
    print(json.dumps({"celulas": len(t), "com_serie": int(ok.sum()), "ativas": int(t.ativa.sum()),
                      "ativas_e_sazonais": int((t.ativa & t.sazonal).sum())}))
    print(t[t.ativa].head(15)[["cell", "geleira", "slope", "saz", "saz_lo", "n_vi_sobe", "razao", "sigma_med"]]
          .to_string())


if __name__ == "__main__":
    main()
