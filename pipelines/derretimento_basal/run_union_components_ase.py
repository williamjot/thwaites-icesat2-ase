"""Componentes do derretimento basal no domínio UNIÃO das variantes da máscara (para o fatorial).

O mapa contínuo do passo 7 só existe no domínio f = 0,9. As variantes f = 0,8 e IC **ampliam** o
domínio perto da linha de aterramento, e testá-las exige m_b também nesses pixels. Aqui as
componentes são recalculadas no domínio união (pixels hidrostáticos em qualquer variante), com
o mesmo método do produto:

- DH/Dt: tendências por célula material (``run_basal_melt_ase.cell_trends``) com os segmentos de
  qualquer variante, depois krigagem por unidade (``run_mb_continuous_ase``: variograma ajustado;
  na Cosgrove, kernel de 8 km, como escolhido pela CV);
- H (firn GEMB e IMAU): mediana por pixel dos segmentos com filtro de presença de gelo,
  lacunas internas preenchidas (``run_flux_term_ase``);
- ∇·u: ITS_LIVE nas escalas 1, 2, 3, 5 km (``data/interim/divergence``);
- SMB: taxa GEMB 2019–2025 (``run_basal_melt_ase.smb_rate_field``);
- distância à frente (IceLines 2019–2024,7) e fração de perda de gelo por pixel;
- distâncias à linha do BedMachine e às linhas de 2019–2025.

Saída: ``data/interim/basal_melt/componentes_uniao_500m.nc``.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
from run_basal_melt_ase import cell_trends, smb_rate_field  # noqa: E402
from run_flux_term_ase import fill_gaps, grid_median  # noqa: E402
from run_mb_continuous_ase import fit_variogram, kernel_predict, krig_predict  # noqa: E402
from run_thickness_ase import ice_presence_filter, speed_split  # noqa: E402
from run_tide_response_lagrangian import advect_to_ref, load_velocity  # noqa: E402

MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
THK = ROOT / "data" / "interim" / "thickness"
DIV = ROOT / "data" / "interim" / "divergence"
DREC = ROOT / "data" / "interim" / "basal_melt" / "dist_LA_recente_500m.nc"
FRONTS = ROOT / "data" / "calving_fronts_densified_500m.parquet"
OUT = ROOT / "data" / "interim" / "basal_melt" / "componentes_uniao_500m.nc"
VARIANTS = ["classe", "classe_f080", "classe_f095", "classe_ic"]
NAMES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}
FRONT_OF = {"Pine_Island": ["PineIsland"], "Thwaites_TEIS": ["Thwaites1", "Thwaites2"],
            "Thwaites_TWGT": ["Thwaites1", "Thwaites2"], "Crosson": ["Crosson"],
            "Dotson": ["Dotson"], "Cosgrove": ["Cosgrove"]}
KERNEL_UNITS = {"Cosgrove": 8e3}


def to_pix(x, y, xs, ys):
    j = np.floor((x - xs[0]) / (xs[1] - xs[0]) + 0.5).astype(np.int64)
    i = np.floor((y - ys[0]) / (ys[1] - ys[0]) + 0.5).astype(np.int64)
    ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
    return i, j, ok


def main() -> int:
    m = xr.open_dataset(MASK)
    xs, ys = m.x.values, m.y.values
    XX, YY = np.meshgrid(xs, ys)
    plat = m.plataforma.values
    union = np.zeros(plat.shape, bool)
    for v in VARIANTS:
        union |= m[v].values == 1

    # unidades no domínio união (Thwaites dividida pela velocidade, como no produto)
    unit = np.full(plat.shape, "", dtype=object)
    for k, n in NAMES.items():
        unit[union & (plat == k)] = n
    ii, jj = np.nonzero(union & (plat == 2))
    fs = speed_split(xs[jj], ys[ii])
    unit[ii[fs], jj[fs]] = "Thwaites_TWGT"
    unit[ii[~fs], jj[~fs]] = "Thwaites_TEIS"

    dh = np.full(plat.shape, np.nan)
    se_dh = np.full(plat.shape, np.nan)
    Hg = np.full(plat.shape, np.nan)
    Hi = np.full(plat.shape, np.nan)
    lost = np.full(plat.shape, np.nan)
    for k, name in NAMES.items():
        cols = ["x", "y", "t", "pass_id", "h_sea", "H_gemb", "H_imau"] + VARIANTS
        d = pd.read_parquet(THK / f"espessura_{name}.parquet", columns=cols)
        anyv = np.zeros(len(d), bool)
        for v in VARIANTS:
            anyv |= d[v].to_numpy() == 1
        d = d[anyv & np.isfinite(d["H_gemb"])].reset_index(drop=True)
        keep = ice_presence_filter(d["x"].to_numpy(), d["y"].to_numpy(), d["h_sea"].to_numpy())
        # perda de gelo por pixel (antes de filtrar)
        i, j, ok = to_pix(d["x"].to_numpy(), d["y"].to_numpy(), xs, ys)
        frac = pd.Series((~keep[ok]).astype(float)).groupby(i[ok] * len(xs) + j[ok]).mean()
        flat = lost.ravel()
        flat[frac.index.to_numpy()] = frac.to_numpy()
        d = d[keep].reset_index(drop=True)
        sel = plat == k
        for arr, col in ((Hg, "H_gemb"), (Hi, "H_imau")):
            g = grid_median(d["x"].to_numpy(), d["y"].to_numpy(), d[col].to_numpy(), xs, ys)
            arr[sel] = g[sel]
        x, y, t = d["x"].to_numpy(), d["y"].to_numpy(), d["t"].to_numpy()
        vel = load_velocity(x, y)
        X0, Y0 = advect_to_ref(x, y, t, vel)
        del vel
        ct = cell_trends(X0, Y0, t, d["H_gemb"].to_numpy(float), d["pass_id"].to_numpy())
        i, j, ok = to_pix(ct["x"].to_numpy() - 250.0, ct["y"].to_numpy() - 250.0, xs, ys)
        ins = ok & union[np.clip(i, 0, len(ys) - 1), np.clip(j, 0, len(xs) - 1)] & \
            (plat[np.clip(i, 0, len(ys) - 1), np.clip(j, 0, len(xs) - 1)] == k)
        dh[i[ins], j[ins]] = ct["dHdt"].to_numpy()[ins]
        se_dh[i[ins], j[ins]] = ct["se"].to_numpy()[ins]
        print(f"{name}: {ins.sum():,} células com DH/Dt no domínio união", flush=True)
        del d, X0, Y0, ct
        gc.collect()
    Hg, Hi = fill_gaps(Hg, union), fill_gaps(Hi, union)

    # krigagem por unidade no domínio união
    dh_c = np.full(plat.shape, np.nan)
    se_c = np.full(plat.shape, np.nan)
    for u in ["Pine_Island", "Thwaites_TEIS", "Thwaites_TWGT", "Crosson", "Dotson", "Cosgrove"]:
        su = unit == u
        ob = su & np.isfinite(dh) & np.isfinite(se_dh)
        xo, yo, vo, so = XX[ob], YY[ob], dh[ob], se_dh[ob]
        if u in KERNEL_UNITS:
            p, s_ = kernel_predict(xo, yo, vo, so, XX[su], YY[su], KERNEL_UNITS[u])
        else:
            vg = fit_variogram(xo, yo, vo, so)
            p, s_ = krig_predict(xo, yo, vo, so, XX[su], YY[su], vg)
        dh_c[su], se_c[su] = p, s_
        print(f"{u}: krigagem em {su.sum():,} pixels a partir de {ob.sum():,} células", flush=True)

    smb, _, _ = smb_rate_field(xs, ys, union)
    drec = xr.open_dataset(DREC).dist_LA_recente_km.values
    # distância às linhas recentes onde o arquivo antigo não cobre (pixels novos da união)
    need = union & ~np.isfinite(drec)
    if need.any():
        sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
        import geopandas as gpd
        from run_open_checks_ase import GL_INSAR, GL_S1, line_points
        gl = gpd.read_file(GL_INSAR)
        pts = np.vstack([line_points(gl[gl.Year.astype(int) >= 2019]),
                         line_points(gpd.read_file(GL_S1).to_crs(3031))])
        drec = drec.copy()
        drec[need] = cKDTree(pts).query(np.c_[XX[need], YY[need]])[0] / 1e3
    fr = pd.read_parquet(FRONTS)
    fr = fr[(fr.epoch_year >= 2019.0) & (fr.epoch_year <= 2025.0)]
    dfront = np.full(plat.shape, np.nan)
    for u, names in FRONT_OF.items():
        su = unit == u
        f = fr[fr.shelf.isin(names)]
        if su.any() and len(f):
            dfront[su] = cKDTree(f[["x", "y"]].to_numpy()).query(np.c_[XX[su], YY[su]])[0] / 1e3

    data = {"DHDt": dh_c, "se_DHDt": se_c, "DHDt_obs": dh, "H_gemb": Hg, "H_imau": Hi,
            "SMB": smb, "perda_gelo_frac": lost, "frente_dist_km": dfront,
            "dist_LA_BedMachine_km": m.dist_gl_km.values, "dist_LA_recente_km": drec}
    for sc in (1, 2, 3, 5):
        data[f"div_{sc}km"] = xr.open_dataset(DIV / f"div_{sc}km.nc").div_media.values
    for v in VARIANTS:
        data[v] = m[v].values
    ds = xr.Dataset({k: (("y", "x"), np.asarray(a, dtype=np.float32)) for k, a in data.items()},
                    coords={"x": xs, "y": ys})
    ds["unidade"] = (("y", "x"), np.array(unit, dtype=str))
    ds["plataforma"] = (("y", "x"), plat)
    ds.to_netcdf(OUT)
    print("gravado", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
