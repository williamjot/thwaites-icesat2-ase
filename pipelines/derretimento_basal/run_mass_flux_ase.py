"""Derretimento basal em Gt/ano por unidade (item 4 da lista de pendências do basal).

Massa = ∫ m_b dA · ρ_i, com m_b em m de gelo eq./a e ρ_i = 917 kg/m³. **Derretimento de gelo
já flutuante não eleva o nível do mar**: o número é água doce entrando no oceano, não
contribuição ao nível do mar.

(a) **Domínio medido:** pixels hidrostáticos (f = 0,9), sem a faixa da frente
    (``front_zone_500m.nc: exclui_frente``).
(b) **Plataforma inteira:** soma (a) com a zona de flexão e a faixa da frente, que não são
    medidas, preenchidas com uma faixa de valores **declarada, não medida**: o limite inferior usa
    a média do domínio medido; o superior, a média dos primeiros 2 km além de d\\* (o derretimento
    é tipicamente maior perto da linha). A área "inteira" é o domínio flutuante da máscara
    (BedMachine ∩ NSIDC-0794 de 2021).

σ: o σ total do envelope (``envelope_por_unidade.csv``, estatístico ⊕ estrutural) aplicado à
média da unidade × área. O σ de (b) soma em quadratura a meia-amplitude do preenchimento.
A TWGT é reportada à parte e não entra no total: 67% da área sai na exclusão da frente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
MB = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
FZ = ROOT / "data" / "interim" / "basal_melt" / "front_zone_500m.nc"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
ENV = ROOT / "outputs" / "basal_melt_ase" / "envelope_por_unidade.csv"
OUT = ROOT / "outputs" / "basal_melt_ase"
RHO_I = 917.0
PIX_M2 = 500.0 * 500.0
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove", "Thwaites_TWGT"]
PLAT = {"Pine_Island": 1, "Thwaites_TEIS": 2, "Thwaites_TWGT": 2, "Crosson": 3, "Dotson": 4,
        "Cosgrove": 5}


def gt(m_mpa, area_m2):
    return m_mpa * area_m2 * RHO_I / 1e12


def main() -> int:
    mb = xr.open_dataset(MB)
    fz = xr.open_dataset(FZ)
    mk = xr.open_dataset(MASK)
    env = pd.read_csv(ENV).set_index("unidade")
    unit = mb.unidade.values
    m = mb.mb.values
    excl = fz.exclui_frente.values == 1
    cls, plat, dist = mk.classe.values, mk.plataforma.values, mk.dist_gl_km.values
    import json
    dstar = {int(k): v for k, v in zip([1, 2, 3, 4, 5], [None] * 5)}
    params = json.loads(mk.classe.attrs["parametros"])
    name_of = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}
    for k, n in name_of.items():
        dstar[k] = params[n]["d_star_km"]["f090"]
    rows = []
    for u in UNITS:
        k = PLAT[u]
        meas = (unit == u) & ~excl & np.isfinite(m)
        a_meas = meas.sum() * PIX_M2
        mean_meas = float(np.mean(m[meas]))
        sig = float(env.loc[u, "sigma_total_mpa"])
        # área inteira da unidade: flutuante da plataforma; TEIS × TWGT pelo rótulo do vizinho
        flo = (cls > 0) & (plat == k)
        if k == 2:
            from scipy.spatial import cKDTree
            XX, YY = np.meshgrid(mk.x.values, mk.y.values)
            lab_src = np.isin(unit, ["Thwaites_TEIS", "Thwaites_TWGT"])
            t = cKDTree(np.c_[XX[lab_src], YY[lab_src]])
            _, idx = t.query(np.c_[XX[flo], YY[flo]])
            lab = unit[lab_src][idx]
            full = np.zeros_like(flo)
            full[flo] = lab == u
        else:
            full = flo
        a_full = full.sum() * PIX_M2
        a_fill = a_full - a_meas
        near = meas & (dist >= dstar[k]) & (dist < dstar[k] + 2.0)
        fill_lo = mean_meas
        fill_hi = float(np.mean(m[near])) if near.any() else mean_meas
        g_meas = gt(mean_meas, a_meas)
        g_lo = g_meas + gt(fill_lo, a_fill)
        g_hi = g_meas + gt(fill_hi, a_fill)
        s_meas = gt(sig, a_meas)
        s_full = np.sqrt(gt(sig, a_full) ** 2 + ((g_hi - g_lo) / 2) ** 2)
        rows.append({"unidade": u, "area_medida_km2": a_meas / 1e6, "area_inteira_km2": a_full / 1e6,
                     "frac_medida": a_meas / a_full, "mb_medio_mpa": mean_meas,
                     "mb_perto_linha_mpa": fill_hi, "Gt_dominio_medido": g_meas,
                     "sigma_Gt_dominio": s_meas, "Gt_inteira_central": (g_lo + g_hi) / 2,
                     "Gt_inteira_min": g_lo, "Gt_inteira_max": g_hi, "sigma_Gt_inteira": s_full})
    t = pd.DataFrame(rows)
    core = t[t.unidade != "Thwaites_TWGT"]
    tot = {"unidade": "TOTAL (sem TWGT)", "area_medida_km2": core.area_medida_km2.sum(),
           "area_inteira_km2": core.area_inteira_km2.sum(),
           "Gt_dominio_medido": core.Gt_dominio_medido.sum(),
           # σ das unidades somado como totalmente correlacionado (o firn e a agregação são
           # escolhas comuns às plataformas): limite superior honesto
           "sigma_Gt_dominio": core.sigma_Gt_dominio.sum(),
           "Gt_inteira_central": core.Gt_inteira_central.sum(),
           "Gt_inteira_min": core.Gt_inteira_min.sum(), "Gt_inteira_max": core.Gt_inteira_max.sum(),
           "sigma_Gt_inteira": core.sigma_Gt_inteira.sum()}
    t = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
    t.to_csv(OUT / "massa_Gt_por_unidade.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
