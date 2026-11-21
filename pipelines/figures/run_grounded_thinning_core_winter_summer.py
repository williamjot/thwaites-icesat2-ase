# -*- coding: utf-8 -*-
"""Gelo aterrado, celula do nucleo de afinamento do mapa regional de dh/dt.

Serie inverno (JJA) x verao (DJF), todos os anos disponiveis, anomalia de massa
acumulada. Reusa o estimador de janela ja validado em
pipelines/figures/_grounded_seasonal_core.py (h_res = h_corr - REMA).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipelines" / "figures"))
import _grounded_seasonal_core as G                        # noqa: E402
from thwaites.corrections.gia import GIAField              # noqa: E402

SCRATCH = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "grounded_H_timeseries" / "nucleo_afinamento_thwaites"
OUT.mkdir(parents=True, exist_ok=True)

JJA = [f"JJA {y}" for y in range(2019, 2026)]
DJF = [f"DJF {y}/{y + 1}" for y in range(2018, 2026)]
WINDOWS = sorted(JJA + DJF, key=lambda w: (int(w[4:8]) + (0.5 if w[:3] == "JJA" else 0.95)))
PARTIAL = {"DJF 2018/2019": "só jan–fev 2019", "DJF 2025/2026": "só dez 2025"}

# o estimador usa TARGET/ORDER como globais do modulo
G.TARGET = tuple(WINDOWS)
G.ORDER = {w: i for i, w in enumerate(WINDOWS)}

XS = (-1467500.0, -1462500.0, -1457500.0)
YS = (-647500.0, -642500.0, -637500.0)
COLS = ["x", "y", "t_year", "mask_class", "track_id", "h_corr", "s_elv",
        "rema", "h_res"]


def load_block():
    keys = {(x, y) for x in XS for y in YS}
    win = (min(XS) - 2500.0, max(XS) + 2500.0, min(YS) - 2500.0, max(YS) + 2500.0)
    parts = []
    for prof in ("jja", "djf"):
        man = json.load(open(ROOT / f"data/{prof}/tiles/manifest.json"))
        sel = [e for e in man if not (e["x_max"] < win[0] or e["x_min"] > win[1]
                                      or e["y_max"] < win[2] or e["y_min"] > win[3])]
        for e in sel:
            t = pq.read_table(
                ROOT / f"data/{prof}/tiles" / e["file"], columns=COLS,
                filters=[("x", ">=", win[0]), ("x", "<", win[1]),
                         ("y", ">=", win[2]), ("y", "<", win[3])]).to_pandas()
            t = t[(t.x >= e["x_min"]) & (t.x < e["x_max"])
                  & (t.y >= e["y_min"]) & (t.y < e["y_max"])]
            t = t[t.mask_class == 2]
            if t.empty:
                continue
            cx, cy = G.cell_center(t.x.to_numpy(), t.y.to_numpy())
            t["cx"], t["cy"] = cx, cy
            t = t[[k in keys for k in zip(t.cx, t.cy)]]
            if t.empty:
                continue
            t["window"] = G.window_labels(t.t_year.to_numpy())
            t = t[t.window.isin(WINDOWS)]
            t["profile"] = prof
            parts.append(t)
            print(f"  [{prof}] {e['file']}: {len(t)}", flush=True)
    obs = pd.concat(parts, ignore_index=True)
    obs = obs[np.isfinite(obs.h_res) & np.isfinite(obs.s_elv) & (obs.s_elv > 0)].copy()
    obs["block"] = obs.profile.map({"jja": 0, "djf": 1}) * 1_000_000 \
        + obs.track_id.astype("int64")
    obs["dx"] = obs.x - obs.cx
    obs["dy"] = obs.y - obs.cy
    return obs


def main():
    cache = ROOT / "data" / "processed" / "grounded_thinning_core_obs.parquet"
    if cache.exists():
        obs = pd.read_parquet(cache)
    else:
        obs = load_block()
        obs.to_parquet(cache, index=False)
    print("observacoes:", len(obs))

    # ---- mapa regional (mesmo arquivo que alimenta mapas_periodo_2019_2025)
    maps = {}
    for prof in ("jja", "djf"):
        n = pd.read_parquet(ROOT / f"data/{prof}/dhdt/dhdt_nodes_qc.parquet")
        maps[prof] = n.set_index(["x", "y"])
    rows = []
    for (cx, cy), g in obs.groupby(["cx", "cy"]):
        sup = g.groupby("window").agg(n=("h_res", "size"), nb=("block", "nunique"))
        sup = sup.reindex(WINDOWS)
        full = WINDOWS if True else None
        core = [w for w in WINDOWS if w not in PARTIAL]
        ok_core = bool((sup.loc[core, "n"] >= G.MIN_OBS).all()
                       and (sup.loc[core, "nb"] >= 2).all())
        mj = maps["jja"].loc[(cx, cy)]
        md = maps["djf"].loc[(cx, cy)]
        rows.append({
            "cx": cx, "cy": cy, "cobertura_completa": ok_core,
            "janelas_ok": int(((sup.n >= G.MIN_OBS) & (sup.nb >= 2)).sum()),
            "min_obs_nucleo": float(sup.loc[core, "n"].min()),
            "min_faixas_nucleo": float(sup.loc[core, "nb"].min()),
            "dhdt_mapa_jja": float(mj.dhdt), "dhdt_mapa_djf": float(md.dhdt),
            "dhdt_mapa_media": float((mj.dhdt + md.dhdt) / 2.0),
            "dist_gl_m": float(mj.dist_gl), "dist_coast_m": float(mj.dist_coast),
            "grounded_frac": float(mj.grounded_frac),
            "mask_class": int(mj.mask_class),
        })
    cand = pd.DataFrame(rows).sort_values("dhdt_mapa_media")
    cand.to_csv(OUT / "candidatas_bloco_3x3.csv", index=False)
    print(cand.to_string())
    elig = cand[cand.cobertura_completa & (cand.mask_class == 2)
                & (cand.grounded_frac >= 0.99) & (cand.dist_gl_m >= 5000)
                & (cand.dist_coast_m >= 2000)]
    if elig.empty:
        raise ValueError("nenhuma celula do bloco com cobertura completa")
    cell = elig.iloc[0]
    print("\nESCOLHIDA:", cell.to_dict())

    sub = obs[(obs.cx == cell.cx) & (obs.cy == cell.cy)].copy()
    ser = G.joint_window_fit(sub, cell.cx, cell.cy, value_col="h_res")
    ser["partial"] = ser.window.isin(PARTIAL.keys())
    ref = ser[ser.window == "JJA 2019"]
    h0 = float(ref.h.iloc[0])
    s0 = float(ref.sigma_h.iloc[0])
    ser["dh_m"] = ser.h - h0
    ser["sigma_dh_m"] = np.where(ser.window == "JJA 2019", 0.0,
                                 np.hypot(ser.sigma_h, s0))
    ser["dM_Gt"] = ser.dh_m * G.GT_PER_M
    ser["sigma_dM_Gt"] = ser.sigma_dh_m * G.GT_PER_M

    # FAC so onde o FDM cobre (termina em 2022,48): quantifica, nao corrige
    fdm = G.FDM(ROOT / "data" / "firn_thwaites.nc")
    years = G.t_year_to_calendar_decimal_year(ser.t_center_year.to_numpy(float))
    fac, dist_fdm, t0, t1 = fdm.sample("fac", float(cell.cx), float(cell.cy), years)
    ser["FAC_m"] = fac

    trends = {}
    for k, season in enumerate(("JJA", "DJF")):
        s = ser[(ser.season == season) & ~ser.partial]
        trends[season] = G.trend(s, "h", "sigma_h", seed=500 + k)
    # desvio sazonal com inclinacao comum: h = a + b t + c [DJF]
    use = ser[~ser.partial]
    tref = float(use.t_center_year.mean())
    X = np.column_stack([np.ones(len(use)), use.t_center_year - tref,
                         (use.season == "DJF").astype(float)])
    w = 1.0 / use.sigma_h.to_numpy() ** 2
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = cov @ (X.T @ (w * use.h.to_numpy()))
    resid = use.h.to_numpy() - X @ beta
    chi2r = float(np.sum(w * resid ** 2) / (len(use) - 3))
    cov_s = cov * max(chi2r, 1.0)
    offset = {"DJF_menos_JJA_m": float(beta[2]),
              "sigma_m": float(np.sqrt(cov_s[2, 2])),
              "slope_comum_m_ano": float(beta[1]),
              "sigma_slope_m_ano": float(np.sqrt(cov_s[1, 1])),
              "chi2_reduzido": chi2r,
              "sigma_inflada_por_chi2": bool(chi2r > 1.0)}
    fac_ok = ser[np.isfinite(ser.FAC_m) & ~ser.partial]
    fac_off = None
    if fac_ok.season.nunique() == 2:
        fac_off = float(fac_ok[fac_ok.season == "DJF"].FAC_m.mean()
                        - fac_ok[fac_ok.season == "JJA"].FAC_m.mean())

    gia = GIAField.from_caron_table(ROOT / "data" / "gia" / "GIA_maps_Caron_et_al_2018")
    vlm, svlm = gia.sample(np.array([ser.lon.iloc[0]]), np.array([ser.lat.iloc[0]]))

    ser.to_csv(OUT / "serie_inverno_verao.csv", index=False, encoding="utf-8")
    report = {
        "celula": {k: (v.item() if hasattr(v, "item") else v) for k, v in cell.to_dict().items()},
        "lon": float(ser.lon.iloc[0]), "lat": float(ser.lat.iloc[0]),
        "fonte_do_mapa": "data/{jja,djf}/dhdt/dhdt_nodes_qc.parquet (mesma de "
                         "outputs/mapas_periodo_2019_2025)",
        "regra_de_escolha": "bloco 3x3 em torno do minimo de dh/dt dos dois mapas; "
                            "vence o dh/dt medio JJA/DJF mais negativo entre as "
                            "celulas aterradas com >=30 obs e >=2 faixas em todas "
                            "as janelas completas",
        "referencia_anomalia": "JJA 2019",
        "Gt_por_metro": G.GT_PER_M,
        "tendencias": trends, "desvio_sazonal": offset,
        "FAC_DJF_menos_JJA_2019_2022_m": fac_off,
        "fdm": {"t_min": t0, "t_max": t1, "dist_celula_m": dist_fdm},
        "gia_mm_ano": float(np.ravel(vlm)[0]) * 1000.0,
        "gia_sigma_mm_ano": float(np.ravel(svlm)[0]) * 1000.0,
        "janelas_parciais": PARTIAL,
    }
    (OUT / "relatorio.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(ser[["window", "t_center_year", "h", "sigma_h", "dh_m", "sigma_dh_m",
               "dM_Gt", "sigma_dM_Gt", "FAC_m", "N_obs", "N_blocks",
               "fit_rmse_window_m", "partial"]].to_string(float_format=lambda v: "%.4f" % v))
    print(json.dumps({"tendencias": trends, "desvio": offset, "fac_off": fac_off,
                      "gia": report["gia_mm_ano"], "fdm": report["fdm"]},
                     indent=1, default=str))
    return ser, report


if __name__ == "__main__":
    main()
