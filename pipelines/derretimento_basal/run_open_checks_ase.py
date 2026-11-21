"""Verificações abertas do derretimento basal (itens 6–8 da lista de pendências).

6. **β baixo além de d\\* (como a queda do Crosson em 2,5–3 km): o gelo toca o leito?**
   Folga = profundidade do leito (BedMachine) − calado (ρi/ρw·H, com H hidrostático medido) −
   (maré baixa ≈ 1 m). Compara-se a folga das células com β significativamente abaixo do
   patamar com a das demais células além de d\\*.
7. **Células do lado "apoiado" com resposta à maré (β − 2σ > 0,3):** estão perto de linhas de
   aterramento recentes (InSAR v2.1 de 2019–2025 e Sentinel-1 de 2025, Andersen)? Se sim, é recuo
   real desde a linha do BedMachine; se não, é mais provável falso positivo.
8. **Manchas de m_b negativo (Dotson, Cosgrove):** fração de pixels com m_b + 2σ < 0, ou seja,
   recongelamento significativo, e se essas manchas se agrupam espacialmente.

Saída: ``outputs/open_checks/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
BETA = ROOT / "outputs" / "tide_response_ase" / "beta_lagr_todas.parquet"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
FLUX = ROOT / "data" / "interim" / "flux_term" / "H_div_500m.nc"
BED = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
MB = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
GL_INSAR = ROOT / "data" / "grounding" / "processed" / "InSAR_GL_ASE_v02.1.gpkg"
GL_S1 = ROOT / "data" / "grounding" / "sources" / "andersen2025_s1" / "S1_1day_GL_2025_ASE" / "S1_1day_GL_2025_ASE.shp"
OUT = ROOT / "outputs" / "open_checks"
RHO = 917.0 / 1027.0


def line_points(gdf, step=200.0):
    pts = []
    for geom in gdf.geometry:
        for ln in getattr(geom, "geoms", [geom]):
            if ln is None or ln.length < 1:
                continue
            for s in np.arange(0, ln.length, step):
                p = ln.interpolate(s)
                pts.append((p.x, p.y))
    return np.array(pts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    b = pd.read_parquet(BETA)
    m = xr.open_dataset(MASK)
    params = json.loads(m.classe.attrs["parametros"])
    fl = xr.open_dataset(FLUX)
    bed = xr.open_dataset(BED)
    X, Y = xr.DataArray(b.x.to_numpy()), xr.DataArray(b.y.to_numpy())
    b["bed"] = bed.bed.interp(x=X, y=Y).values
    b["H"] = fl.H_gemb.interp(x=X, y=Y, method="nearest").values
    b["folga_m"] = (-b.bed) - RHO * b.H - 1.0          # profundidade do leito − calado − maré baixa
    rows = []
    for s, g in b.groupby("shelf"):
        P, Plo = params[s]["P"], params[s]["P_ic95"][0]
        ds = params[s]["d_star_km"]["f090"]
        far = g[(g.dist_gl_km >= ds) & (g.dist_gl_km <= 15) & np.isfinite(g.folga_m)]
        low = far[(far.se_beta < 0.25) & (far.beta + 2 * far.se_beta < Plo)]
        rest = far.drop(low.index)
        rows.append({"plataforma": s, "n_beta_baixo": len(low), "n_resto": len(rest),
                     "folga_mediana_beta_baixo_m": float(low.folga_m.median()) if len(low) else np.nan,
                     "folga_mediana_resto_m": float(rest.folga_m.median()),
                     "frac_beta_baixo_folga_lt_50m": float((low.folga_m < 50).mean()) if len(low) else np.nan,
                     "frac_resto_folga_lt_50m": float((rest.folga_m < 50).mean())})
        if s == "Crosson":
            dip = g[(g.dist_gl_km >= 2.5) & (g.dist_gl_km < 3.0)]
            rows[-1]["crosson_2p5_3km_folga_mediana_m"] = float(dip.folga_m.median())
            rows[-1]["crosson_2p5_3km_beta_mediano"] = float(dip.beta.median())
    t6 = pd.DataFrame(rows)
    t6.to_csv(OUT / "6_beta_baixo_folga.csv", index=False)

    # 7: células apoiadas com resposta à maré × linhas de aterramento recentes
    gl = gpd.read_file(GL_INSAR)
    recent = gl[gl.Year.astype(int) >= 2019]
    pts = line_points(recent)
    try:
        s1 = gpd.read_file(GL_S1).to_crs(3031)
        pts = np.vstack([pts, line_points(s1)])
    except Exception as e:  # noqa: BLE001
        print("S1 2025 não lido:", e)
    old = gl[gl.Year.astype(int) <= 2011]
    tree_new, tree_old = cKDTree(pts), cKDTree(line_points(old))
    gr = b[b.dist_gl_km < 0]
    sig = gr[gr.beta - 2 * gr.se_beta > 0.3]
    ctrl = gr.drop(sig.index).sample(min(len(sig) * 5, len(gr) - len(sig)), random_state=0)
    rows = []
    for lab, g in (("resposta_significativa", sig), ("controle", ctrl)):
        dn = tree_new.query(np.c_[g.x, g.y])[0] / 1e3
        do = tree_old.query(np.c_[g.x, g.y])[0] / 1e3
        for s, gg in g.assign(dn=dn, do=do).groupby("shelf"):
            rows.append({"grupo": lab, "plataforma": s, "n": len(gg),
                         "dist_mediana_GL_2019_2025_km": float(gg.dn.median()),
                         "frac_a_menos_de_1km_GL_recente": float((gg.dn < 1).mean()),
                         "dist_mediana_GL_ate_2011_km": float(gg.do.median())})
    t7 = pd.DataFrame(rows)
    t7.to_csv(OUT / "7_apoiado_com_mare.csv", index=False)

    # 8: recongelamento significativo
    mb = xr.open_dataset(MB)
    mm, se, u = mb.mb.values, mb.se_mb.values, mb.unidade.values
    rows = []
    for un in ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]:
        s = (u == un) & np.isfinite(mm) & np.isfinite(se)
        neg = s & (mm + 2 * se < 0)
        lab, n = ndimage.label(neg)
        sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
        rows.append({"unidade": un, "frac_mb_negativo": float((mm[s] < 0).mean()),
                     "frac_recongelamento_signif": float(neg[s].mean()),
                     "n_manchas_signif": int(n), "maior_mancha_km2": float(sizes.max() * 0.25),
                     "manchas_ge_5km2": int((sizes * 0.25 >= 5).sum())})
    t8 = pd.DataFrame(rows)
    t8.to_csv(OUT / "8_recongelamento.csv", index=False)
    pd.set_option("display.width", 250)
    for t in (t6, t7, t8):
        print(t.round(2).to_string(index=False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
