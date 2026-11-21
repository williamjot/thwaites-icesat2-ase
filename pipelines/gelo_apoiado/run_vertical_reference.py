"""Define explicitamente ``h_surface``, nível do mar e ``freeboard``.

Entrada esperada: pontos ATL06 já limitados a gelo flutuante dinâmico e com a
predição CATS no campo ``tide_ocean``. ``dem_h`` não participa do cálculo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.qc.grounding_zone import FLOATING_CONFIDENT


def main():
    parser = argparse.ArgumentParser(description="Referencial vertical ATL06")
    parser.add_argument("--profile", default="anual")
    parser.add_argument("--input", default="atl06_shelf_dynamic.parquet")
    parser.add_argument("--output", default="atl06_shelf_freeboard.parquet")
    parser.add_argument("--mdt", default="vertical_reference/dtu15mdt_ase.nc")
    parser.add_argument("--tide-model-sigma-m", type=float, default=0.05)
    parser.add_argument("--dac-sigma-m", type=float, default=0.02)
    parser.add_argument("--geoid-sigma-m", type=float, default=0.10)
    parser.add_argument("--mdt-sigma-m", type=float, default=0.05)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="vertical_reference")
    source = cfg.paths.interim / args.input
    destination = cfg.paths.interim / args.output
    mdt_path = cfg.paths.data_dir / args.mdt
    if not source.exists() or not mdt_path.exists():
        raise FileNotFoundError(f"entradas ausentes: {source}, {mdt_path}")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.interpolate import RegularGridInterpolator
    import xarray as xr

    mdt_dataset = xr.open_dataset(mdt_path)
    mdt_lat = np.asarray(mdt_dataset.lat.values, float)
    mdt_lon = np.asarray(mdt_dataset.lon.values, float)
    mdt_values = np.asarray(mdt_dataset.mdt.values, float)
    mdt_dataset.close()
    mdt_interp = RegularGridInterpolator(
        (mdt_lat, mdt_lon), mdt_values, method="linear",
        bounds_error=False, fill_value=np.nan)

    bedmachine = next(iter(sorted(
        cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))), None)
    if bedmachine is None:
        raise FileNotFoundError("BedMachine Antarctica ausente")
    bed = xr.open_dataset(bedmachine)
    bx, by = np.asarray(bed.x.values, float), np.asarray(bed.y.values, float)
    ix = np.where((bx >= -2.0e6) & (bx <= -1.2e6))[0]
    iy = np.where((by >= -0.9e6) & (by <= 0.0))[0]
    geoid = np.asarray(bed.geoid.isel(x=ix, y=iy).values, float)
    gx, gy = bx[ix], by[iy]
    bed.close()
    if gy[0] > gy[-1]:
        gy, geoid = gy[::-1], geoid[::-1, :]
    geoid_interp = RegularGridInterpolator(
        (gy, gx), geoid, method="linear", bounds_error=False,
        fill_value=np.nan)

    parquet = pq.ParquetFile(source)
    required = {
        "h_elv", "s_elv", "sigma_geo_h", "tide_ocean", "dac",
        "tide_ocean_source", "lon", "lat", "x", "y", "grounding_state"}
    if missing := required.difference(parquet.schema_arrow.names):
        raise ValueError(f"entrada não satisfaz contrato vertical: {sorted(missing)}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    writer = None
    totals = {"n": 0, "complete": 0, "cats": 0}
    try:
        for batch in parquet.iter_batches(batch_size=250_000):
            frame = batch.to_pandas()
            h_surface = (frame.h_elv.to_numpy(float)
                         - frame.tide_ocean.to_numpy(float)
                         - frame.dac.to_numpy(float))
            geoid_local = geoid_interp(np.c_[frame.y, frame.x])
            mdt_local = mdt_interp(np.c_[frame.lat, frame.lon])
            mean_sea_surface = geoid_local + mdt_local
            freeboard = h_surface - mean_sea_surface
            h_surface_sigma = np.sqrt(
                frame.s_elv.to_numpy(float) ** 2
                + frame.sigma_geo_h.to_numpy(float) ** 2
                + args.tide_model_sigma_m ** 2 + args.dac_sigma_m ** 2)
            freeboard_sigma = np.sqrt(
                h_surface_sigma ** 2 + args.geoid_sigma_m ** 2
                + args.mdt_sigma_m ** 2)
            cats = frame.tide_ocean_source.to_numpy() == 1
            floating = frame.grounding_state.to_numpy() == FLOATING_CONFIDENT
            complete = (cats & floating & np.isfinite(h_surface)
                        & np.isfinite(mean_sea_surface) & np.isfinite(freeboard)
                        & np.isfinite(freeboard_sigma))
            frame["h_surface"] = h_surface.astype(np.float32)
            frame["geoid_eigen6c4"] = geoid_local.astype(np.float32)
            frame["mean_dynamic_topography"] = mdt_local.astype(np.float32)
            frame["mean_sea_surface"] = mean_sea_surface.astype(np.float32)
            frame["freeboard"] = np.where(complete, freeboard, np.nan).astype(np.float32)
            frame["h_surface_sigma"] = h_surface_sigma.astype(np.float32)
            frame["freeboard_sigma"] = np.where(
                complete, freeboard_sigma, np.nan).astype(np.float32)
            frame["vertical_reference_complete"] = complete
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    partial, table.schema, compression="zstd")
            writer.write_table(table)
            totals["n"] += len(frame)
            totals["complete"] += int(complete.sum())
            totals["cats"] += int(cats.sum())
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("entrada vazia")
    partial.replace(destination)

    report = {
        "STATUS": "VERTICAL_REFERENCE_EXPLICIT",
        "input": str(source), "output": str(destination), **totals,
        "definitions": {
            "h_surface": "h_li - CATS_ocean_tide - DAC, WGS84 ellipsoid",
            "mean_sea_surface": "BedMachine EIGEN-6C4 geoid + DTU15MDT",
            "freeboard": "h_surface - mean_sea_surface",
            "dem_h": "QC/context only; never used as freeboard",
        },
        "already_applied_in_h_li": [
            "solid Earth tide", "ocean load tide", "pole tide",
            "neutral atmospheric delay", "first photon bias",
            "transmit pulse shape bias"],
        "not_reapplied": ["tide_earth", "tide_load", "tide_pole",
                          "neutat_delay_total"],
        "equilibrium_tide": (
            "ATL06 tide_equilibrium not added: CATS2008_v2023 contains mf/mm "
            "and pyTMD inference includes long-period constituents"),
        "uncertainty_priors_m": {
            "tide_model": args.tide_model_sigma_m, "DAC": args.dac_sigma_m,
            "geoid": args.geoid_sigma_m, "MDT": args.mdt_sigma_m},
        "uncertainty_caveat": (
            "geoid/MDT/tide values are sensitivity priors pending local "
            "open-ocean/crossover validation; they are not vendor error bars"),
    }
    report_path = cfg.paths.tables / "vertical_reference_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    log.info(f"Referencial vertical -> {destination}")
    log.info(f"Completos: {totals['complete']:,}/{totals['n']:,}")


if __name__ == "__main__":
    main()
