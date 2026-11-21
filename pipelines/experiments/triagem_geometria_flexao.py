"""Triagem geométrica para inversão de flexão por maré: trilha ICESat-2 × fluxo × linha de aterramento.

Para cada ponto da linha de aterramento (InSAR GL v2.1, densificada a 500 m), mede:
- ângulo entre a direção da trilha ICESat-2 (azimute real medido em ATL06) e o fluxo (ITS_LIVE);
- velocidade e deslocamento em 91 dias (um ciclo de repetição);
- distância ao trecho de linha de aterramento mais próximo de OUTRA parte da linha
  (proxy de topografia confinante: margens, ilhas, pontos de ancoragem) dentro de 15 km.
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd, xarray as xr, glob
from shapely.geometry import Point
from scipy.spatial import cKDTree

ROOT = sys.argv[1]
gl = gpd.read_file(f"{ROOT}/data/grounding/processed/InSAR_GL_ASE_v02.1.gpkg")
mask = xr.open_dataset(f"{ROOT}/data/interim/ase_shelf_mask_240m.nc")["shelf"]
names = {int(k): v for k, v in (p.split("=") for p in mask.attrs["legenda"].split("; ")) if int(k) > 0}

# pontos de GL densificados
pts = []
for geom in gl.geometry:
    for line in getattr(geom, "geoms", [geom]):
        if line.length < 1:
            continue
        for s in np.arange(0, line.length, 500.0):
            p = line.interpolate(s); pts.append((p.x, p.y))
P = np.unique(np.round(np.array(pts), -2), axis=0)

# plataforma: rótulo mais frequente num raio de 3 km a jusante (pixels flutuantes próximos)
mx, my, mv = mask["x"].values, mask["y"].values, mask.values
ii, jj = np.nonzero(mv > 0)
tree_m = cKDTree(np.c_[mx[jj], my[ii]])
d, idx = tree_m.query(P, distance_upper_bound=3000)
ok = np.isfinite(d)
P, lab = P[ok], mv[ii[idx[ok]], jj[idx[ok]]]

# velocidade
v = xr.open_dataset(f"{ROOT}/data/velocity_thwaites.nc")
vx = v["VX"].interp(x=xr.DataArray(P[:, 0]), y=xr.DataArray(P[:, 1])).values
vy = v["VY"].interp(x=xr.DataArray(P[:, 0]), y=xr.DataArray(P[:, 1])).values
spd = np.hypot(vx, vy)

# azimute real das trilhas: direção local de segmentos ATL06 (um ciclo de SON)
dirs = []
for f in sorted(glob.glob(f"{ROOT}/data/ase_v2/son/ATL06_2019*.parquet"))[:120]:
    t = pd.read_parquet(f, columns=["x", "y", "beam", "x_atc"])
    for _, g in t.groupby("beam"):
        g = g.sort_values("x_atc")
        if len(g) < 50:
            continue
        xy = g[["x", "y"]].to_numpy()[::25]
        dv = np.diff(xy, axis=0)
        dirs.append(np.c_[xy[:-1] + dv / 2, dv / np.linalg.norm(dv, axis=1)[:, None]])
D = np.vstack(dirs)
_, k = cKDTree(D[:, :2]).query(P)
tdir = D[k, 2:]
cosang = np.abs((vx * tdir[:, 0] + vy * tdir[:, 1]) / spd)
ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))          # 0 = trilha paralela ao fluxo

# confinamento: BedMachine indica rocha/gelo apoiado de 3 a 15 km a jusante, ao longo do fluxo
bm = xr.open_dataset(f"{ROOT}/data/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")["mask"]
f = np.c_[vx, vy] / np.maximum(spd, 1e-9)[:, None]
conf = np.zeros(len(P), bool)
for dist in np.arange(3000, 15001, 500):
    q = P + f * dist
    m = bm.interp(x=xr.DataArray(q[:, 0]), y=xr.DataArray(q[:, 1]), method="nearest").values
    conf |= np.isin(m, [1, 2, 4])

df = pd.DataFrame({"shelf": [names[int(l)] for l in lab], "ang": ang, "spd": spd,
                   "desl91_km": spd * 91 / 365.25 / 1e3, "conf": conf})
df = df[np.isfinite(df.spd) & (df.spd > 10)]
df["apto"] = (df.ang < 30) & ~df.conf
r = df.groupby("shelf").agg(n_pts_GL=("ang", "size"),
                            ang_mediano=("ang", "median"),
                            frac_ang30=("ang", lambda a: (a < 30).mean()),
                            vel_mediana=("spd", "median"),
                            desl91_km_med=("desl91_km", "median"),
                            frac_confinada=("conf", "mean"),
                            frac_apta=("apto", "mean"))
pd.set_option("display.width", 200)
print(r.round(2).to_string())
print("apto E velocidade < 1000 m/a:",
      df.assign(lento=df.spd < 1000).groupby("shelf").apply(lambda g: (g.apto & g.lento).mean()).round(3).to_dict())
