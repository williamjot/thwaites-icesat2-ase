"""
Extrai ATL11 v007 para a faixa da GL em Crosson/Dotson (config/gz_fh_etapa2b_atl11.yaml). DADO NOVO.
Um .h5 por vez em pasta temporária, apagado no finally. Só JJA + DJF. Retomável.
Credenciais: EARTHDATA_USERNAME / EARTHDATA_PASSWORD no ambiente (nunca gravadas em disco).
Saída: data/atl11/crosson_dotson/<granulo>.parquet
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import earthaccess
import h5py
import numpy as np
import pandas as pd
import pyogrio
import yaml
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "gz_fh_etapa2b_atl11.yaml"
ADD = ROOT / "data/external_geometry/ADD_v7_6/add_coastline_high_res_line_v7_6.gpkg"
RES = 250.0
EPOCH = np.datetime64("2018-01-01T00:00:00")


def main(granule_csv: str) -> dict:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["data"]["output_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    months = set(cfg["selection"]["seasons_months"])
    leg = pd.read_parquet(ROOT / "data/processed/legacy_basal_final_nodes_jja.parquet")
    reg = leg[leg.shelf.isin(["Thwaites", "Crosson", "Dotson"])]
    x0, x1, y0, y1 = reg.x.min() - 30e3, reg.x.max() + 30e3, reg.y.min() - 30e3, reg.y.max() + 30e3
    add = pyogrio.read_dataframe(ADD, layer="add_coastline_high_res_line_v7_6", bbox=(x0, y0, x1, y1))
    gla = add[add.surface == "grounding line"]
    nx, ny = int((x1 - x0) / RES) + 1, int((y1 - y0) / RES) + 1
    D = distance_transform_edt(features.rasterize(((g, 1) for g in gla.geometry.values), out_shape=(ny, nx),
                               transform=from_origin(x0, y1, RES, RES), all_touched=True, dtype="uint8") == 0) * RES / 1000
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    earthaccess.login(strategy="environment")

    gl = pd.read_csv(granule_csv)
    log = []
    t0 = time.time()
    for k, row in enumerate(gl.itertuples(), 1):
        base = row.name.split(".h5")[0]
        out = out_dir / f"{base}.parquet"
        if out.exists():
            log.append(dict(granule=base, status="ja_extraido")); continue
        tmp = tempfile.mkdtemp(prefix="atl11_")
        try:
            res = earthaccess.search_data(short_name="ATL11", version="007", granule_name=base + ".h5")
            if not res:
                log.append(dict(granule=base, status="nao_encontrado")); continue
            fp = earthaccess.download(res[:1], local_path=tmp, show_progress=False)[0]
            parts = []
            with h5py.File(fp, "r") as f:
                for pt in ("pt1", "pt2", "pt3"):
                    if pt not in f:
                        continue
                    g = f[pt]
                    lat = g["latitude"][:]; lon = g["longitude"][:]
                    x, y = tf.transform(lon, lat)
                    inb = (x > x0) & (x < x1) & (y > y0) & (y < y1)
                    if not inb.any():
                        continue
                    dd = np.full(x.size, np.inf)
                    dd[inb] = D[((y1 - y[inb]) / RES).astype(int).clip(0, ny - 1), ((x[inb] - x0) / RES).astype(int).clip(0, nx - 1)]
                    keep = dd <= 15.0
                    if not keep.any():
                        continue
                    idx = np.flatnonzero(keep)
                    cyc = g["cycle_number"][:]
                    h = g["h_corr"][:][idx]; hs = g["h_corr_sigma"][:][idx]; dt = g["delta_time"][:][idx]
                    q = g["quality_summary"][:][idx]
                    tide = g["cycle_stats/tide_ocean"][:][idx]; dac = g["cycle_stats/dac"][:][idx]
                    xatc = g["ref_surf/x_atc"][:][idx]; fq = g["ref_surf/fit_quality"][:][idx]; rp = g["ref_pt"][:][idx]
                    n_pt, n_c = h.shape
                    ii, jj = np.meshgrid(np.arange(n_pt), np.arange(n_c), indexing="ij")
                    ii, jj = ii.ravel(), jj.ravel()
                    df = pd.DataFrame(dict(pt=int(pt[-1]), ref_pt=rp[ii], x_atc=xatc[ii], x=x[idx][ii], y=y[idx][ii],
                                           dist_add_km=dd[idx][ii], fit_quality=fq[ii], cycle=cyc[jj],
                                           delta_time=dt.ravel(), h_corr=h.ravel(), h_corr_sigma=hs.ravel(),
                                           quality_summary=q.ravel(), tide_ocean=tide.ravel(), dac=dac.ravel()))
                    df = df[np.isfinite(df.h_corr) & (np.abs(df.h_corr) < 1e30) & np.isfinite(df.delta_time) & (np.abs(df.delta_time) < 1e30)]
                    tt = EPOCH + (df.delta_time.to_numpy() * 1e6).astype("timedelta64[us]")
                    mon = pd.DatetimeIndex(tt).month
                    df = df[np.isin(mon, list(months))].assign(time=tt[np.isin(mon, list(months))])
                    parts.append(df)
            if parts:
                P = pd.concat(parts, ignore_index=True).assign(rgt=int(row.rgt), granule=base)
                P.to_parquet(out, index=False)
                log.append(dict(granule=base, status="extraido", n_rows=len(P)))
            else:
                pd.DataFrame().to_parquet(out, index=False)
                log.append(dict(granule=base, status="sem_pontos_na_faixa"))
        except Exception as e:  # registra e segue
            log.append(dict(granule=base, status="erro", error=f"{type(e).__name__}: {str(e)[:200]}"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"[{k}/{len(gl)}] {base} {log[-1]['status']} {time.time() - t0:.0f}s", flush=True)
    summ = pd.DataFrame(log).status.value_counts().to_dict()
    (out_dir / "EXTRACTION_LOG.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    return summ


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), ensure_ascii=False))
