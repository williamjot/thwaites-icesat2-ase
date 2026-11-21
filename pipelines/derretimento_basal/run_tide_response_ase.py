"""Resposta medida da superfície à maré em torno da linha de aterramento (passo 3 do plano basal).

Pergunta: onde o gelo das plataformas da ASE sobe e desce 100% com a maré (equilíbrio
hidrostático, onde a espessura por flutuação vale) e onde a resposta é parcial (zona de flexão)?

Modelo por célula de 500 m, com todas as passagens 2019–2025 das quatro estações:

    h = a + gx·dx + gy·dy + c·(t − t̄) + s·sin2πt + k·cos2πt + β·T

- ``h``: ``h_elv`` do ATL06 (marés terrestre, de carga e polar já aplicadas; oceânica e DAC não);
- ``T`` = maré oceânica CATS2008-v2023 (extrapolada até 10 km além da grade, sem o fator de
  flexão do modelo) + ``dac``. Usar o ``tide_ocean`` do ATL06 é impossível: ele é nulo sobre o
  gelo que o ATL06 considera apoiado (32% dos pontos num grânulo conferido), justamente onde a
  pergunta está;
- o plano local (gx, gy) é obrigatório: as passagens repetidas não caem no mesmo lugar, e sem o
  plano o resíduo topográfico engole a maré;
- a tendência e o ciclo anual absorvem o afinamento e a variação sazonal do firn, que de outro
  modo poderiam se confundir com o alias da maré na amostragem de 91 dias.

β ≈ 1: flutuação livre. β < 1: flexão ou apoio. Erro de β: variância robusta por agrupamento
(CR1), com grupo = passagem ``(rgt, cycle)``. Os segmentos de uma passagem compartilham erro de
maré, apontamento e atmosfera e não são independentes.

Nenhum limiar de classificação é fixado aqui. O script mede β e o seu erro; a regra de "zona
hidrostática" será escolhida depois, olhando a distribuição.

Etapas (com cache em ``data/interim/tide_response/``):
1. faixa: segmentos com QC == 0, altura sobre o nível do mar ≥ 5 m (exclui mar e gelo marinho),
   a até 15 km da linha de aterramento do BedMachine v4.1, rotulados por plataforma;
2. maré CATS + DAC para cada segmento, por plataforma, em lotes;
3. ajuste por célula de 500 m.

Saída: ``outputs/tide_response_ase/beta_celulas.parquet`` e um resumo por plataforma e faixa de
distância.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thwaites import load_config  # noqa: E402
from thwaites.corrections.cats_tide import CatsTidePredictor  # noqa: E402

STORE = ROOT / "data" / "ase_v2"
SHELF_MASK = ROOT / "data" / "interim" / "ase_shelf_mask_240m.nc"
BEDMACHINE = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
MDT = ROOT / "data" / "vertical_reference" / "dtu15mdt_ase.nc"
CACHE = ROOT / "data" / "interim" / "tide_response"
OUT = ROOT / "outputs" / "tide_response_ase"
BAND_M = 15_000.0
CELL = 500.0
SHELVES = ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"]   # Getz fora (cobertura)
COLS = ["x", "y", "lon", "lat", "h_elv", "t_year", "rgt", "cycle", "dac", "geoid",
        "atl06_quality_summary"]


def band_grid():
    """Distância assinada à linha de aterramento do BedMachine (+ flutuante, − apoiado) e rótulo
    de plataforma, numa grade de 500 m."""
    m = xr.open_dataset(SHELF_MASK)["shelf"]
    legend = dict(p.split("=") for p in m.attrs["legenda"].split("; "))
    lab_of = {v: int(k) for k, v in legend.items() if int(k) > 0}
    x0, x1 = float(m.x.min()) - BAND_M, float(m.x.max()) + BAND_M
    y0, y1 = float(m.y.min()) - BAND_M, float(m.y.max()) + BAND_M
    bm = xr.open_dataset(BEDMACHINE)["mask"].sel(x=slice(x0, x1), y=slice(y1, y0))
    mk = bm.values
    flo, grd = mk == 3, np.isin(mk, [1, 2, 4])
    d_in_flo = ndimage.distance_transform_edt(flo) * 500.0      # distância até o não-flutuante
    d_in_grd = ndimage.distance_transform_edt(grd) * 500.0
    signed = np.where(flo, d_in_flo, np.where(grd, -d_in_grd, np.nan))
    # rótulo: plataforma mais próxima (máscara de 240 m subamostrada a ~1 km)
    sm = m.values[::4, ::4]
    ii, jj = np.nonzero(sm > 0)
    tree = cKDTree(np.c_[m.x.values[::4][jj], m.y.values[::4][ii]])
    XX, YY = np.meshgrid(bm.x.values, bm.y.values)
    sel = np.isfinite(signed) & (np.abs(signed) <= BAND_M)
    dd, kk = tree.query(np.c_[XX[sel], YY[sel]], distance_upper_bound=BAND_M + 5000)
    lab = np.zeros(mk.shape, np.int8)
    lab_sel = np.zeros(sel.sum(), np.int8)
    ok = np.isfinite(dd)
    lab_sel[ok] = sm[ii[kk[ok]], jj[kk[ok]]]
    lab[sel] = lab_sel
    keep_ids = [lab_of[s] for s in SHELVES]
    lab[~np.isin(lab, keep_ids)] = 0
    names = {lab_of[s]: s for s in SHELVES}
    return bm.x.values, bm.y.values, signed, lab, names


def stage1(grid):
    xs, ys, signed, lab, names = grid
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    mdt = xr.open_dataset(MDT)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "faixa.parquet"
    writer = None
    n_tot = 0
    files = sorted(STORE.glob("*/*.parquet"))
    for n, f in enumerate(files, 1):
        d = pd.read_parquet(f, columns=COLS)
        d = d[(d["atl06_quality_summary"] == 0) & np.isfinite(d["h_elv"])
              & (d["h_elv"] < 1e30) & np.isfinite(d["dac"]) & (np.abs(d["dac"]) < 1e30)
              & np.isfinite(d["geoid"]) & (np.abs(d["geoid"]) < 1e30)]
        if len(d):
            j = np.rint((d["x"].to_numpy() - xs[0]) / dx).astype(int)
            i = np.rint((d["y"].to_numpy() - ys[0]) / dy).astype(int)
            ok = (i >= 0) & (i < lab.shape[0]) & (j >= 0) & (j < lab.shape[1])
            d, i, j = d[ok], i[ok], j[ok]
            lb, sd = lab[i, j], signed[i, j]
            keep = lb > 0
            d, lb, sd = d[keep], lb[keep], sd[keep]
        if len(d):
            m = mdt["mdt"].interp(lat=xr.DataArray(d["lat"].to_numpy()),
                                  lon=xr.DataArray(d["lon"].to_numpy())).values
            hs = d["h_elv"].to_numpy() - d["geoid"].to_numpy() - np.nan_to_num(m)
            keep = hs >= 5.0
            t = pa.table({
                "x": d["x"].to_numpy()[keep], "y": d["y"].to_numpy()[keep],
                "lon": d["lon"].to_numpy()[keep], "lat": d["lat"].to_numpy()[keep],
                "h": d["h_elv"].to_numpy()[keep].astype(np.float32),
                "t": d["t_year"].to_numpy()[keep],
                "pass_id": (d["rgt"].to_numpy()[keep].astype(np.int32) * 1000
                            + d["cycle"].to_numpy()[keep].astype(np.int32)),
                "dac": d["dac"].to_numpy()[keep].astype(np.float32),
                "dist_gl": sd[keep].astype(np.float32),
                "shelf": lb[keep].astype(np.int8)})
            if t.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(out, t.schema, compression="zstd")
                writer.write_table(t)
                n_tot += t.num_rows
        if n % 500 == 0:
            print(f"  faixa: {n}/{len(files)} arquivos, {n_tot:,} segmentos", flush=True)
    if writer is not None:
        writer.close()
    print(f"faixa: {n_tot:,} segmentos", flush=True)


def stage2(cfg, names):
    src = pq.ParquetFile(CACHE / "faixa.parquet")
    with CatsTidePredictor(cfg) as pred:
        for k, name in names.items():
            out = CACHE / f"mare_{name}.parquet"
            if out.is_file():
                continue
            tab = pq.read_table(CACHE / "faixa.parquet", filters=[("shelf", "=", k)])
            d = tab.to_pandas()
            del tab
            tide = np.empty(len(d), np.float32)
            step = int(cfg.cats.chunk_size)
            for a in range(0, len(d), step):
                b = a + step
                tide[a:b] = pred.predict(d["lon"].to_numpy()[a:b], d["lat"].to_numpy()[a:b],
                                         d["t"].to_numpy()[a:b])
            d["tide_cats"] = tide
            d.drop(columns=["lon", "lat"]).to_parquet(out, index=False, compression="zstd")
            print(f"maré {name}: {len(d):,} segmentos, {np.isnan(tide).mean():.1%} sem maré",
                  flush=True)
            del d, tide
            gc.collect()
    del src


def fit_cell(g):
    t = g["t"].to_numpy()
    T = (g["tide_cats"].to_numpy(float) + g["dac"].to_numpy(float))
    X = np.c_[np.ones(len(g)), g["x"].to_numpy() - g["x"].mean(),
              g["y"].to_numpy() - g["y"].mean(), t - t.mean(),
              np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), T]
    y = g["h"].to_numpy(float)
    XtX = X.T @ X
    try:
        XtXi = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    b = XtXi @ X.T @ y
    r = y - X @ b
    # CR1: soma dos escores por passagem
    pid = g["pass_id"].to_numpy()
    u, inv = np.unique(pid, return_inverse=True)
    S = np.zeros((len(u), X.shape[1]))
    np.add.at(S, inv, X * r[:, None])
    G, n, p = len(u), len(y), X.shape[1]
    corr = G / (G - 1) * (n - 1) / (n - p)
    V = corr * XtXi @ (S.T @ S) @ XtXi
    # amplitude informativa da maré entre passagens (média por passagem)
    Tp = np.bincount(inv, weights=T) / np.bincount(inv)
    return b[6], float(np.sqrt(max(V[6, 6], 0))), float(np.std(Tp)), G, n, float(r.std())


def stage3(names, min_pass=10):
    rows = []
    for k, name in names.items():
        d = pd.read_parquet(CACHE / f"mare_{name}.parquet")
        d = d[np.isfinite(d["tide_cats"])]
        d["ci"] = np.floor(d["x"] / CELL).astype(np.int64)
        d["cj"] = np.floor(d["y"] / CELL).astype(np.int64)
        npass = d.groupby(["ci", "cj"])["pass_id"].nunique()
        good = npass[npass >= min_pass].index
        d = d.set_index(["ci", "cj"]).loc[good].reset_index()
        for (ci, cj), g in d.groupby(["ci", "cj"], sort=False):
            res = fit_cell(g)
            if res is None:
                continue
            beta, se, sdT, G, n, rms = res
            rows.append({"shelf": name, "x": (ci + 0.5) * CELL, "y": (cj + 0.5) * CELL,
                         "dist_gl_km": float(g["dist_gl"].median()) / 1e3,
                         "beta": beta, "se_beta": se, "sd_tide_m": sdT,
                         "n_pass": G, "n_seg": n, "rms_m": rms})
        print(f"ajuste {name}: {sum(r['shelf'] == name for r in rows):,} células", flush=True)
        del d
        gc.collect()
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "beta_celulas.parquet", index=False)
    return out


def summarize(df):
    bins = [-15, -5, -2, -1, 0, 1, 2, 5, 10, 15]
    df = df.assign(faixa=pd.cut(df["dist_gl_km"], bins))
    s = (df.groupby(["shelf", "faixa"], observed=True)
           .agg(n_cel=("beta", "size"), beta_med=("beta", "median"),
                se_med=("se_beta", "median"), sdT_med=("sd_tide_m", "median"))
           .round(3))
    s.to_csv(OUT / "resumo_beta_por_distancia.csv")
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, default=0, help="0 = todas; 1, 2 ou 3")
    args = ap.parse_args()
    cfg = load_config()
    grid = band_grid()
    names = grid[4]
    if args.stage in (0, 1) and not (CACHE / "faixa.parquet").is_file():
        stage1(grid)
    if args.stage in (0, 2):
        stage2(cfg, names)
    if args.stage in (0, 3):
        summarize(stage3(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
