# -*- coding: utf-8 -*-
"""Driver do relatorio de gelo aterrado."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _grounded_seasonal_core as G          # noqa: E402
import _grounded_seasonal_figs as F            # noqa: E402
from thwaites.corrections.gia import GIAField  # noqa: E402

ROOT = G.ROOT
OUT = G.OUT


def main():
    chosen, pool = G.select_cells()
    print("pool de celulas aterradas elegiveis:", len(pool))
    for k, v in chosen.items():
        print("  %-14s (%.0f, %.0f)  dist_GL=%.1f km  min_tracks=%d  "
              "min_obs=%d  dhdt_nodes=%+.2f" %
              (k, v.cx, v.cy, v.dist_gl / 1000.0, v.min_tracks_window,
               v.min_obs_window, v.dhdt))

    cache = ROOT / "data" / "processed" / "grounded_seasonal_obs.parquet"
    if cache.exists():
        obs = pd.read_parquet(cache)
        print("observacoes do cache:", len(obs))
    else:
        obs = G.load_observations(list(chosen.values()))
        obs.to_parquet(cache, index=False)
        print("observacoes carregadas dos tiles:", len(obs))

    fdm = G.FDM(ROOT / "data" / "firn_thwaites.nc")
    gia_field = GIAField.from_caron_table(
        ROOT / "data" / "gia" / "GIA_maps_Caron_et_al_2018")

    series_by_role = {}
    info_by_role = {}
    metrics = {}
    metrics_ice = {}
    gia_by_role = {}
    fdm_audit = {}
    for i, (role, cell) in enumerate(chosen.items()):
        sub = obs[(obs.cx == cell.cx) & (obs.cy == cell.cy)].copy()
        ser = G.joint_window_fit(sub, cell.cx, cell.cy)
        ser, aud = G.enrich(ser, fdm, float(cell.cx), float(cell.cy))
        vlm, sigma_vlm = gia_field.sample(
            np.array([ser.lon.iloc[0]]), np.array([ser.lat.iloc[0]]))
        gia_by_role[role] = {
            "vlm_m_per_year": float(np.ravel(vlm)[0]),
            "vlm_mm_per_year": float(np.ravel(vlm)[0]) * 1000.0,
            "sigma_vlm_mm_per_year": float(np.ravel(sigma_vlm)[0]) * 1000.0,
        }
        series_by_role[role] = ser
        fdm_audit[role] = aud
        info_by_role[role] = {
            "cell_x": float(cell.cx), "cell_y": float(cell.cy),
            "lon": float(ser.lon.iloc[0]), "lat": float(ser.lat.iloc[0]),
            "dist_gl_m": float(cell.dist_gl),
            "dist_coast_m": float(cell.dist_coast),
            "grounded_fraction": float(cell.grounded_frac),
            "dhdt_node_product_m_per_year": float(cell.dhdt),
            "dhdt_node_product_sigma_jack": float(cell.dhdt_err_jack),
        }
        metrics[role] = G.trend(ser, "h", "sigma_h", seed=1000 + i)
        metrics_ice[role] = G.trend(ser, "h_ice", "sigma_h_ice",
                                    seed=2000 + i)
        print("  %-14s dh/dt = %+.3f +- %.3f m/ano (p=%.4f) | gelo-eq %+.3f"
              % (role, metrics[role]["slope_m_per_year"],
                 metrics[role]["slope_sigma_m_per_year"],
                 metrics[role]["slope_p_two_sided"],
                 metrics_ice[role]["slope_m_per_year"]))

    primary = series_by_role["primary"]
    tr = G.transitions(primary)

    primary.to_csv(OUT / "grounded_cell_por_janela.csv", index=False,
                   encoding="utf-8")
    tr.to_csv(OUT / "grounded_cell_por_transicao.csv", index=False,
              encoding="utf-8")
    pd.concat([s.assign(role=r) for r, s in series_by_role.items()],
              ignore_index=True).to_csv(
        OUT / "grounded_transecto_por_janela.csv", index=False,
        encoding="utf-8")

    F.figure_main(primary, tr, metrics["primary"], metrics_ice["primary"],
                  info_by_role["primary"], gia_by_role["primary"],
                  OUT / "grounded_cell_dhdt_massa",
                  G.GT_PER_M, G.GT_PER_MM_SLE)

    order = ["perto_GL", "primary", "intermediario", "interior"]
    order = [r for r in order if r in series_by_role]
    nice = {"perto_GL": "Mais perto da GL", "primary": "Primária",
            "intermediario": "Intermediária", "interior": "Interior"}
    rows = []
    for r in sorted(order, key=lambda k: info_by_role[k]["dist_gl_m"]):
        rows.append({
            "role": r,
            "label": "%s\n%.0f km" % (nice[r],
                                      info_by_role[r]["dist_gl_m"] / 1000.0),
            "slope_m_per_year": metrics[r]["slope_m_per_year"],
            "slope_sigma_m_per_year": metrics[r]["slope_sigma_m_per_year"],
            "slope_p_two_sided": metrics[r]["slope_p_two_sided"],
        })
    F.figure_transect(rows, OUT / "grounded_transecto_tendencia_massa",
                      G.GT_PER_M)

    tend = metrics["primary"]
    tend_ice = metrics_ice["primary"]
    vlm_mm = gia_by_role["primary"]["vlm_mm_per_year"]
    report = {
        "status": "COMPLETE_REAL_DATA_ONLY",
        "produto": "ICESat-2 ATL06 (tiles do recorte de gelo aterrado)",
        "dominio": "gelo aterrado: BedMachine mask==2, buffer 5 km da linha de "
                   "aterramento e 2 km da costa (config/default.yaml: grounded)",
        "grandeza": "h_corr (elevacao de superficie). A relacao hidrostatica "
                    "NAO se aplica em gelo aterrado; nao ha amplificacao "
                    "rho_w/(rho_w-rho_i).",
        "contrato_vertical": "mare e DAC NAO subtraidas (so valem sobre "
                             "flutuante); geoide irrelevante porque so "
                             "diferencas temporais entram.",
        "estimador_janela": {
            "modelo": "um intercepto por janela + plano local comum dx/dy",
            "robustez": "Huber IRLS",
            "pesos": "1/s_elv^2 normalizado por bloco (track_id x perfil)",
            "sigma": "max(SE cluster-robusta sobre blocos, RMSE/sqrt(N_blocos))",
            "bloco": "track_id (passagem along-track por feixe)",
            "minimo_observacoes": G.MIN_OBS,
            "sem_interpolacao_temporal": True,
        },
        "massa": {
            "formula": "dM = rho_i * A * (dh - dFAC)",
            "rho_i_kg_m3": G.RHO_I,
            "area_m2": G.AREA_M2,
            "Gt_por_metro": G.GT_PER_M,
            "Gt_por_mm_SLE": G.GT_PER_MM_SLE,
            "FAC": "GSFC-FDM v1.2.1 (data/firn_thwaites.nc), amostrado "
                   "co-temporalmente; fill -9999 mascarado",
        },
        "celulas": info_by_role,
        "fdm_audit": fdm_audit,
        "gia_caron2018": gia_by_role,
        "tendencia_h": metrics,
        "tendencia_gelo_equivalente": metrics_ice,
        "primary_resumo": {
            "dh_dt_m_ano": tend["slope_m_per_year"],
            "sigma_m_ano": tend["slope_sigma_m_per_year"],
            "ci95": [tend["slope_ci95_low"], tend["slope_ci95_high"]],
            "p": tend["slope_p_two_sided"],
            "dh_ice_dt_m_ano": tend_ice["slope_m_per_year"],
            "dM_dt_Gt_ano": tend_ice["slope_m_per_year"] * G.GT_PER_M,
            "sigma_dM_dt_Gt_ano": (tend_ice["slope_sigma_m_per_year"]
                                   * G.GT_PER_M),
            "SLE_mm_ano": -(tend_ice["slope_m_per_year"] * G.GT_PER_M)
                          / G.GT_PER_MM_SLE,
            "dh_dt_corrigido_GIA_m_ano": (tend["slope_m_per_year"]
                                          - vlm_mm / 1000.0),
            "GIA_vlm_mm_ano": vlm_mm,
        },
    }
    with open(OUT / "grounded_timeseries_report.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=str)

    pd.set_option("display.width", 260)
    print()
    print(primary[["window", "t_center_year", "h", "sigma_h", "FAC", "h_ice",
                   "dh_cum_m", "dM_cum_Gt", "N_obs", "N_blocks"]]
          .to_string(float_format=lambda v: "%.4f" % v))
    print()
    print(tr[["de", "para", "dt_anos", "dh_m", "sigma_dh_m", "dhdt_m_ano",
              "sigma_dhdt_m_ano", "dFAC_m", "dh_ice_m", "dM_Gt",
              "sigma_dM_Gt", "SLE_mm", "z"]]
          .to_string(float_format=lambda v: "%+.4f" % v))
    print()
    print(json.dumps(report["primary_resumo"], indent=1))
    print()
    print("FDM audit:", json.dumps(fdm_audit["primary"], indent=1))
    print("GIA:", json.dumps(gia_by_role["primary"], indent=1))


if __name__ == "__main__":
    main()
