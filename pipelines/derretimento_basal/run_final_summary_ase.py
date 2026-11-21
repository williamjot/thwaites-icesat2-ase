"""Tabela final do derretimento basal por unidade (m/a e Gt/a), com o domínio e a incerteza finais.

- Domínio e valor central: produto v2 (``mb_final_v2_500m.nc``, de ``run_front_epoch_ase.py``:
  frente por data; Thwaites com a correção β·maré + radar), média por área no domínio.
- Incerteza: ``incerteza_v2_resumo.csv`` (``run_uncertainty_v2_ase.py``), calculada no MESMO
  domínio: σ estrutural = √(fatorial p05–p95² + agregação² + velocidade²); σ estatístico da série
  v2. A tabela falha se a média do mapa não bater com a do arquivo de incerteza.
- **Zona não medida** (zona de flexão/aterramento ativa e faixa da frente): a área inteira é o
  flutuante da máscara (BedMachine ∩ NSIDC-0794 de 2021). Faixa declarada para o derretimento
  ali: do mínimo (média do domínio medido) ao máximo (média dos primeiros 2 km do domínio medido
  junto ao corte, onde o derretimento é tipicamente maior). Não é medida.
- Total sem TWGT: σ das unidades somado linearmente (escolhas de método comuns a todas as
  plataformas).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "outputs"
MB = ROOT / "data" / "interim" / "basal_melt" / "mb_final_v2_500m.nc"     # produto central v2
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
DREC = ROOT / "data" / "interim" / "basal_melt" / "dist_LA_recente_500m.nc"
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]
PLAT = {"Pine_Island": 1, "Thwaites_TEIS": 2, "Crosson": 3, "Dotson": 4, "Cosgrove": 5}
OPT = {"Pine_Island": 3, "Thwaites_TEIS": 2, "Crosson": 2, "Dotson": 2, "Cosgrove": 2}
RHO_I, PIX = 917.0, 0.25e6


def main() -> int:
    mb = xr.open_dataset(MB)
    m, u = mb.mb.values, mb.unidade.values
    dom = mb.dominio.values == 1
    mk = xr.open_dataset(MASK)
    cls, plat, dbm = mk.classe.values, mk.plataforma.values, mk.dist_gl_km.values
    drec = xr.open_dataset(DREC).dist_LA_recente_km.values
    unc = pd.read_csv(O / "basal_melt_ase" / "incerteza_v2_resumo.csv").set_index("unidade")
    rows = []
    for un in UNITS:
        k = dom & (u == un) & np.isfinite(m)
        mean = float(np.mean(m[k]))
        dv = float(unc.loc[un, "velocidade"])
        s_struct = float(unc.loc[un, "sigma_estrutural"])
        s_stat = float(unc.loc[un, "sigma_estat"])
        s_tot = float(unc.loc[un, "sigma_total"])
        assert abs(unc.loc[un, "mb_mapa_mpa"] - mean) < 1e-3, "incerteza de outro domínio"
        # área inteira da unidade
        flo = (cls > 0) & (plat == PLAT[un])
        if un == "Thwaites_TEIS":
            flo = flo & ((u == "Thwaites_TEIS") | ((u == "") & False))
            from scipy.spatial import cKDTree
            XX, YY = np.meshgrid(mk.x.values, mk.y.values)
            src = np.isin(u, ["Thwaites_TEIS", "Thwaites_TWGT"])
            allflo = (cls > 0) & (plat == 2)
            _, idx = cKDTree(np.c_[XX[src], YY[src]]).query(np.c_[XX[allflo], YY[allflo]])
            lab = u[src][idx]
            flo = np.zeros_like(allflo)
            flo[allflo] = lab == un
        a_meas, a_full = k.sum() * PIX, flo.sum() * PIX
        near = k & (np.minimum(dbm, drec) < np.nanpercentile(np.minimum(dbm, drec)[k], 5) + 2.0)
        fill_lo, fill_hi = sorted([mean, float(np.mean(m[near]))])
        g_meas = mean * a_meas * RHO_I / 1e12
        g_lo = g_meas + fill_lo * (a_full - a_meas) * RHO_I / 1e12
        g_hi = g_meas + fill_hi * (a_full - a_meas) * RHO_I / 1e12
        rows.append({"unidade": un, "mb_mpa": mean, "sigma_estat": s_stat, "sigma_estrutural": s_struct,
                     "sigma_total": s_tot, "fator_velocidade": dv,
                     "area_medida_km2": a_meas / 1e6, "area_inteira_km2": a_full / 1e6,
                     "frac_medida": a_meas / a_full,
                     "Gt_medido": g_meas, "sigma_Gt_medido": s_tot * a_meas * RHO_I / 1e12,
                     "Gt_inteira_min": g_lo, "Gt_inteira_max": g_hi,
                     "sigma_Gt_inteira": float(np.hypot(s_tot * a_full * RHO_I / 1e12, (g_hi - g_lo) / 2))})
    t = pd.DataFrame(rows)
    tot = {"unidade": "TOTAL (sem TWGT)", "Gt_medido": t.Gt_medido.sum(),
           "sigma_Gt_medido": t.sigma_Gt_medido.sum(), "Gt_inteira_min": t.Gt_inteira_min.sum(),
           "Gt_inteira_max": t.Gt_inteira_max.sum(), "sigma_Gt_inteira": t.sigma_Gt_inteira.sum(),
           "area_medida_km2": t.area_medida_km2.sum(), "area_inteira_km2": t.area_inteira_km2.sum()}
    t = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
    t.to_csv(O / "basal_melt_ase" / "TABELA_FINAL_derretimento_basal.csv", index=False)
    pd.set_option("display.width", 260)
    print(t.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
