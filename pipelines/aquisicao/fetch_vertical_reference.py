"""Recorta DTU15MDT no ASE sem preservar a grade global em disco.

DTU15MDT usa o geoide EIGEN-6C4, compatível com a variável ``geoid`` do
BedMachine Antarctica v4.1. O arquivo global compactado é verificado, lido em
streaming e apagado depois que o pequeno recorte NetCDF é gravado.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging

URL = "https://ftp.space.dtu.dk/pub/DTU15/2_MIN/DTU15MDT_2min.zip"
SHA256 = "25073f49a8f5787d4d1778cc93c47d7c11c80930e1caf91f3ba4db001274493f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(path: Path, log):
    if path.exists() and _sha256(path) == SHA256:
        log.info(f"reutilizando {path.name} com SHA256 verificado")
        return
    path.unlink(missing_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    urllib.request.urlretrieve(URL, partial)
    if _sha256(partial) != SHA256:
        partial.unlink(missing_ok=True)
        raise IOError("SHA256 inválido para DTU15MDT")
    partial.replace(path)


def _tokens(stream):
    for line in stream:
        for token in line.split():
            yield float(token)


def _extract(zip_path: Path, destination: Path, cfg, buffer_deg: float, log):
    lon_min = (cfg.roi.lon_min - buffer_deg) % 360.0
    lon_max = (cfg.roi.lon_max + buffer_deg) % 360.0
    lat_min = cfg.roi.lat_min - buffer_deg
    lat_max = cfg.roi.lat_max + buffer_deg
    with zipfile.ZipFile(zip_path) as archive:
        entry = archive.infolist()[0]
        with archive.open(entry) as binary, io.TextIOWrapper(
                binary, encoding="ascii") as text:
            header = [float(value) for value in text.readline().split()]
            if len(header) != 6:
                raise ValueError(f"cabeçalho GRAVSOFT inesperado: {header}")
            south, north, west, east, dy, dx = header
            nlat = int(round((north - south) / dy)) + 1
            nlon = int(round((east - west) / dx)) + 1
            lat_axis = south + np.arange(nlat) * dy
            lon_axis = west + np.arange(nlon) * dx
            rows = np.flatnonzero((lat_axis >= lat_min) & (lat_axis <= lat_max))
            cols = np.flatnonzero((lon_axis >= lon_min) & (lon_axis <= lon_max))
            if not len(rows) or not len(cols):
                raise ValueError("ROI não intercepta DTU15MDT")
            values = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
            iterator = _tokens(text)
            row_lookup = {int(row): i for i, row in enumerate(rows)}
            col_set = set(map(int, cols))
            for row in range(int(rows[-1]) + 1):
                target_row = row_lookup.get(row)
                for col in range(nlon):
                    value = next(iterator)
                    if target_row is not None and col in col_set:
                        target_col = int(col - cols[0])
                        values[target_row, target_col] = value

    lon = lon_axis[cols]
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lat = lat_axis[rows]
    from pyproj import Transformer
    longitude, latitude = np.meshgrid(lon, lat)
    x, y = Transformer.from_crs(4326, 3031, always_xy=True).transform(
        longitude, latitude)

    from netCDF4 import Dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with Dataset(partial, "w", format="NETCDF4") as dataset:
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.source_url = URL
        dataset.source_model = "DTU15MDT 2 arc-minute"
        dataset.geoid_reference = "EIGEN-6C4"
        dataset.vertical_definition = "mean sea surface minus EIGEN-6C4 geoid"
        dataset.license = "source distributed by DTU Space"
        for name, data, dimensions, units in (
                ("lat", lat, ("lat",), "degrees_north"),
                ("lon", lon, ("lon",), "degrees_east"),
                ("x", x, ("lat", "lon"), "m"),
                ("y", y, ("lat", "lon"), "m"),
                ("mdt", values, ("lat", "lon"), "m")):
            variable = dataset.createVariable(
                name, "f4" if name == "mdt" else "f8", dimensions,
                zlib=True, complevel=4)
            variable[:] = data
            variable.units = units
        dataset["mdt"].long_name = "mean dynamic topography above EIGEN-6C4 geoid"
    partial.replace(destination)
    log.info(
        f"DTU15MDT -> {destination} | {values.shape}, "
        f"mediana {np.nanmedian(values):+.3f} m")


def main():
    parser = argparse.ArgumentParser(description="Recorta DTU15MDT para ASE")
    parser.add_argument("--profile", default="anual")
    parser.add_argument("--buffer-deg", type=float, default=1.0)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="fetch_vertical_reference")
    temporary = cfg.paths.data_dir / "vertical_reference_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    zip_path = temporary / "DTU15MDT_2min.zip"
    destination = (cfg.paths.data_dir / "vertical_reference"
                   / "dtu15mdt_ase.nc")
    try:
        _download(zip_path, log)
        _extract(zip_path, destination, cfg, args.buffer_deg, log)
    finally:
        zip_path.unlink(missing_ok=True)
        if temporary.exists() and not any(temporary.iterdir()):
            temporary.rmdir()


if __name__ == "__main__":
    main()
