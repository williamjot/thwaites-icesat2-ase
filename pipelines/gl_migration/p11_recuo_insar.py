"""InSAR + ICESat-2, passo B: recuo da linha de aterramento por InSAR ao longo de linhas de fluxo.

Linhas InSAR: MEaSUREs InSAR-Based Grounding Lines v2.1 (NSIDC-0498), recorte ASE em
data/grounding/processed/InSAR_GL_ASE_v02.1.gpkg, mais Sentinel-1 2025 (Andersen et al. 2025, 1 dia).
Cada linha InSAR resulta de pares de interferogramas: a data usada é a média de Date_1..Date_4.
Linhas de fluxo: velocidade média ITS_LIVE v2 anual 2019–2025 (120 m), integração RK2 de 250 m,
25 km para montante e 15 km para jusante de sementes espaçadas de 2 km sobre as linhas InSAR de
2019–2021 (as do meio do período ICESat-2).
Posição s: distância ao longo da linha de fluxo (positiva a jusante) até o cruzamento com cada linha
InSAR; com mais de um cruzamento, o mais próximo da semente (≤ 12 km).
Recuo: inclinação de Theil–Sen de s contra a data (m/ano), em 2016–2025 e em 2019–2025; IC por
bootstrap das datas. A dispersão entre linhas do mesmo mês (ICEYE diário de 2024) mede a migração
com a maré e o erro do InSAR, e entra na incerteza.
Saída: outputs/gl_migration/fluxo_linhas.parquet (vértices), recuo_insar_cruzamentos.parquet,
recuo_insar_por_linha.parquet
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import theilslopes
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
RNG = np.random.default_rng(20261117)
PASSO, MONT, JUS, ESP = 250.0, 25000.0, 15000.0, 2000.0
GRUPO = {"Pine Island": "Pine Island", "Thwaites": "Thwaites", "Pope Kohler Smith": "Smith-Pope-Kohler",
         "Smith Kohler": "Smith-Pope-Kohler", "SmithKohler": "Smith-Pope-Kohler",
         "Pope Smith Kohler": "Smith-Pope-Kohler", "Pope": "Smith-Pope-Kohler", "Haynes": "Haynes"}


def linhas_insar():
    gl = gpd.read_file(ROOT / "data" / "grounding" / "processed" / "InSAR_GL_ASE_v02.1.gpkg")
    gl = gl[gl.Glac_Name.isin(GRUPO)].copy()
    gl["geleira"] = gl.Glac_Name.map(GRUPO)
    datas = gl[["Date_1", "Date_2", "Date_3", "Date_4"]].apply(pd.to_datetime, errors="coerce")
    gl["data"] = datas.mean(axis=1)
    gl["ano_dec"] = gl.data.dt.year + (gl.data.dt.dayofyear - 0.5) / 365.25
    gl = gl[gl.data.dt.year >= 2016]
    s1 = gpd.read_file(ROOT / "data" / "grounding" / "sources" / "andersen2025_s1" / "S1_1day_GL_2025_ASE" /
                       "S1_1day_GL_2025_ASE.shp").to_crs(3031)
    s1 = s1.assign(Sensor="Sentinel-1 (Andersen 2025)", geleira="todas",
                   data=pd.Timestamp("2025-07-01"), ano_dec=2025.5)
    return gl.explode(index_parts=False), s1.explode(index_parts=False)


def campo_velocidade():
    v = xr.open_dataset(ROOT / "data" / "velocity_itslive_annual.nc")
    vx = v.vx.mean("time", skipna=True).values
    vy = v.vy.mean("time", skipna=True).values
    x, y = v.x.values, v.y.values
    if y[0] > y[-1]:
        y, vx, vy = y[::-1], vx[::-1], vy[::-1]
    f = dict(bounds_error=False, fill_value=np.nan)
    return RegularGridInterpolator((y, x), vx, **f), RegularGridInterpolator((y, x), vy, **f)


def traca(p0, fvx, fvy, dist, sentido):
    pts = [np.asarray(p0, float)]
    andado = 0.0
    while andado < dist:
        p = pts[-1]
        u = np.array([fvx((p[1], p[0])), fvy((p[1], p[0]))], float).ravel()
        sp = np.hypot(*u)
        if not np.isfinite(sp) or sp < 5:
            break
        pm = p + sentido * 0.5 * PASSO * u / sp
        um = np.array([fvx((pm[1], pm[0])), fvy((pm[1], pm[0]))], float).ravel()
        spm = np.hypot(*um)
        if not np.isfinite(spm) or spm < 5:
            break
        pts.append(p + sentido * PASSO * um / spm)
        andado += PASSO
    return np.array(pts)


def main():
    gl, s1 = linhas_insar()
    fvx, fvy = campo_velocidade()
    sementes = []
    for geleira, g in gl[gl.data.dt.year.between(2019, 2021)].groupby("geleira"):
        for geom in g.geometry:
            n = int(geom.length // ESP)
            for k in range(n + 1):
                pt = geom.interpolate(k * ESP)
                sementes.append((geleira, pt.x, pt.y))
    sem = pd.DataFrame(sementes, columns=["geleira", "x", "y"])
    # sementes muito próximas (linhas repetidas do mesmo trecho): uma a cada 1,5 km
    sem["ix"], sem["iy"] = (sem.x // 1500).astype(int), (sem.y // 1500).astype(int)
    sem = sem.drop_duplicates(["geleira", "ix", "iy"]).reset_index(drop=True)
    print("sementes", sem.groupby("geleira").size().to_dict())

    vert, cruz = [], []
    todas = pd.concat([gl[["geleira", "Sensor", "data", "ano_dec", "geometry"]],
                       s1[["geleira", "Sensor", "data", "ano_dec", "geometry"]]], ignore_index=True)
    sidx = todas.sindex
    for i, r in sem.iterrows():
        up = traca((r.x, r.y), fvx, fvy, MONT, -1)[::-1]
        dn = traca((r.x, r.y), fvx, fvy, JUS, +1)[1:]
        if len(up) < 20 or len(dn) < 10:
            continue
        P = np.vstack([up, dn])
        s = (np.arange(len(P)) - (len(up) - 1)) * PASSO
        vert.append(pd.DataFrame({"fid": i, "geleira": r.geleira, "s": s, "x": P[:, 0], "y": P[:, 1]}))
        ls = LineString(P)
        for j in sidx.query(ls, predicate="intersects"):
            row = todas.iloc[j]
            if row.geleira not in (r.geleira, "todas"):
                continue
            inter = ls.intersection(row.geometry)
            pts = [inter] if inter.geom_type == "Point" else [q for q in getattr(inter, "geoms", []) if q.geom_type == "Point"]
            if not pts:
                continue
            ss = [ls.project(q) - (len(up) - 1) * PASSO for q in pts]
            k = int(np.argmin(np.abs(ss)))
            if abs(ss[k]) <= 12000:
                cruz.append(dict(fid=i, geleira=r.geleira, sensor=row.Sensor, data=row.data,
                                 ano_dec=row.ano_dec, s=ss[k], n_cruz=len(pts)))
    vert = pd.concat(vert, ignore_index=True)
    cruz = pd.DataFrame(cruz)
    vert.to_parquet(OUT / "fluxo_linhas.parquet", index=False)
    cruz.to_parquet(OUT / "recuo_insar_cruzamentos.parquet", index=False)

    res = []
    for fid, g in cruz.groupby("fid"):
        for ini in (2016, 2019):
            q = g[g.ano_dec >= ini]
            # um valor por mês (mediana): evita que as séries diárias do ICEYE dominem o ajuste
            q = q.assign(mes=q.data.dt.to_period("M")).groupby("mes").agg(ano=("ano_dec", "median"), s=("s", "median")).reset_index()
            if q.ano.nunique() < 3 or q.ano.max() - q.ano.min() < 3:
                continue
            sl, ic, lo, hi = theilslopes(q.s, q.ano)
            bs = []
            for _ in range(500):
                b = q.iloc[RNG.integers(0, len(q), len(q))]
                if b.ano.nunique() >= 3:
                    bs.append(theilslopes(b.s, b.ano)[0])
            res.append(dict(fid=fid, geleira=g.geleira.iloc[0], inicio=ini, n_meses=len(q),
                            t0=q.ano.min(), t1=q.ano.max(), recuo_m_ano=sl,
                            lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                            s_medio=q.s.median(),
                            disp_mesmo_mes_m=g.assign(mes=g.data.dt.to_period("M")).groupby("mes").s.std().median()))
    res = pd.DataFrame(res)
    res.to_parquet(OUT / "recuo_insar_por_linha.parquet", index=False)
    print(res.groupby(["geleira", "inicio"]).agg(n=("fid", "size"), mediana=("recuo_m_ano", "median"),
                                                 p25=("recuo_m_ano", lambda v: v.quantile(.25)),
                                                 p75=("recuo_m_ano", lambda v: v.quantile(.75)),
                                                 disp=("disp_mesmo_mes_m", "median")).round(0))


if __name__ == "__main__":
    main()
