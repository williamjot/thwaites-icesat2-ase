"""Baixa e recorta FAC/SMB co-temporais para o balanço basal.

Produtos:
  * IMAU-FDM v1.2A FirnAir, 27 km, 10 dias, 1979--2023;
  * RACMO2.3p2 SMB total, 27 km, mensal, 1979--2022.

Somente o recorte ASE em EPSG:3031 é preservado. Os NetCDF antárticos são
arquivos temporários, têm checksum verificado e são removidos em ``finally``.
"""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging


FILES = {
    "grid": {
        "name": "Height_latlon_ANT27.nc",
        "url": "https://zenodo.org/api/records/7845736/files/Height_latlon_ANT27.nc/content",
        "md5": "1a2d519603689c078c9e197529aeacd1",
        "size": 4029604,
    },
    "smb": {
        "name": "smb_monthlyS_ANT27_ERA5-3H_RACMO2.3p2_197901_202212.nc",
        "url": "https://zenodo.org/api/records/7845736/files/smb_monthlyS_ANT27_ERA5-3H_RACMO2.3p2_197901_202212.nc/content",
        "md5": "825353623bbbaf898ea979faae4d53f9",
        "size": 30238818,
    },
    "fac": {
        "name": "FDMv12A_FirnAir_ANT27_1979-2023_detrended.nc",
        "url": "https://zenodo.org/api/records/14634258/files/FDMv12A_FirnAir_ANT27_1979-2023_detrended.nc/content",
        "md5": "2a04e6f44d1626dc531d495e592a7d10",
        "size": 827535507,
    },
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(item: dict[str, str], temporary: Path, log) -> Path:
    path = temporary / item["name"]
    if path.exists() and _md5(path) == item["md5"]:
        log.info(f"reutilizando temporário verificado: {path.name}")
        return path
    expected_size = int(item["size"])
    prefix_size = path.stat().st_size if path.exists() else 0
    if prefix_size >= expected_size:
        path.unlink(missing_ok=True)
        prefix_size = 0
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    # Zenodo limita a vazão por conexão, mas suporta HTTP ranges. Preserva um
    # prefixo interrompido e baixa o restante em partes independentes.
    workers = 8 if expected_size > 100_000_000 else 2
    remaining = expected_size - prefix_size
    part_size = int(np.ceil(remaining / workers))

    def fetch_part(index_start_end):
        index, start, end = index_start_end
        part = temporary / f"{item['name']}.part{index:02d}"
        request = urllib.request.Request(
            item["url"], headers={"Range": f"bytes={start}-{end}"})
        last_error = None
        for _ in range(5):
            try:
                with (
                    urllib.request.urlopen(request, timeout=120) as response,
                    part.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
                if part.stat().st_size != end - start + 1:
                    raise IOError(f"parte {index} tem tamanho inválido")
                return part
            except Exception as error:  # repetição de falhas transitórias HTTP
                last_error = error
                part.unlink(missing_ok=True)
        raise IOError(f"falha ao baixar parte {index}: {last_error}")

    ranges = []
    for index in range(workers):
        start = prefix_size + index * part_size
        end = min(expected_size - 1, start + part_size - 1)
        if start <= end:
            ranges.append((index, start, end))
    log.info(
        f"baixando {item['name']}: prefixo {prefix_size:,} bytes + "
        f"{len(ranges)} partes")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(fetch_part, ranges))
    with partial.open("wb") as output:
        if prefix_size:
            with path.open("rb") as prefix:
                shutil.copyfileobj(prefix, output, length=8 * 1024 * 1024)
        for part in parts:
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
            part.unlink()
    path.unlink(missing_ok=True)
    if _md5(partial) != item["md5"]:
        partial.unlink(missing_ok=True)
        raise IOError(f"checksum MD5 inválido: {item['name']}")
    partial.replace(path)
    return path


def _decimal_year(dates) -> np.ndarray:
    return np.asarray([
        date.year + (date.timetuple().tm_yday - 1
                     + (date.hour + date.minute / 60) / 24) /
        (366 if calendar.isleap(date.year) else 365)
        for date in dates
    ], dtype=float)


def _write_subset(source: Path, output: Path, field_candidates: tuple[str, ...],
                  time_min: float, spatial_mask: np.ndarray, rows: slice,
                  cols: slice, x: np.ndarray, y: np.ndarray, lon: np.ndarray,
                  lat: np.ndarray, *, output_variable: str, source_doi: str,
                  log) -> None:
    from netCDF4 import Dataset, num2date

    with Dataset(source) as src:
        name = next((candidate for candidate in field_candidates
                     if candidate in src.variables), None)
        if name is None:
            raise ValueError(
                f"nenhuma variável {field_candidates} em {source.name}; "
                f"disponíveis={list(src.variables)}")
        time_var = src["time"]
        raw_time = np.asarray(time_var[:])
        units = str(getattr(time_var, "units", "")).lower()
        if "since" in units:
            dates = num2date(raw_time, time_var.units,
                            calendar=getattr(time_var, "calendar", "standard"),
                            only_use_cftime_datetimes=False)
            time = _decimal_year(np.atleast_1d(dates))
        elif "year" in units or np.nanmedian(raw_time) > 1900:
            time = raw_time.astype(float)
        else:
            raise ValueError(f"eixo temporal não reconhecido em {source.name}: {units}")
        keep_time = np.flatnonzero(time >= time_min)
        if not len(keep_time):
            raise ValueError(f"{source.name} não cobre t >= {time_min}")

        variable = src[name]
        index = [keep_time]
        for dimension in variable.dimensions[1:]:
            if dimension.lower() in {"height", "level", "lev"}:
                index.append(0)
            elif dimension in {"rlat", "y"}:
                index.append(rows)
            elif dimension in {"rlon", "x"}:
                index.append(cols)
            else:
                raise ValueError(
                    f"dimensão não tratada {dimension!r} em {source.name}")
        values = np.ma.filled(variable[tuple(index)], np.nan)
        values = np.asarray(values, dtype=np.float32).reshape(len(keep_time), -1)
        values = values[:, spatial_mask.ravel()]
        source_units = str(getattr(variable, "units", ""))

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with Dataset(partial, "w", format="NETCDF4") as dst:
        dst.createDimension("time", len(keep_time))
        dst.createDimension("point", int(spatial_mask.sum()))
        dst.source_file = source.name
        dst.source_doi = source_doi
        dst.crs = "EPSG:3031"
        dst.spatial_sampling = "native RACMO/IMAU-FDM rotated grid, about 27 km"
        dst.extrapolation_policy = "forbidden"
        tv = dst.createVariable("time", "f8", ("time",))
        tv[:] = time[keep_time]
        tv.units = "decimal year"
        for key, array, units_out in (
                ("x", x[spatial_mask], "m"), ("y", y[spatial_mask], "m"),
                ("lon", lon[spatial_mask], "degrees_east"),
                ("lat", lat[spatial_mask], "degrees_north")):
            var = dst.createVariable(key, "f8", ("point",), zlib=True)
            var[:] = array
            var.units = units_out
        data = dst.createVariable(
            output_variable, "f4", ("time", "point"), zlib=True,
            complevel=4, fill_value=np.float32(9.96921e36))
        data[:] = values
        data.units = source_units
        if output_variable == "smb_monthly":
            data.long_name = "monthly cumulative total surface mass balance"
            data.sign_convention = "positive mass gain at surface"
        else:
            data.long_name = "firn air content"
    partial.replace(output)
    log.info(
        f"recorte -> {output} ({len(keep_time)} épocas, "
        f"{int(spatial_mask.sum())} células)")


def main() -> None:
    parser = argparse.ArgumentParser(description="FAC/SMB para basal melt ASE")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--time-min", type=float, default=2018.5)
    parser.add_argument("--buffer-deg", type=float, default=1.0)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="fetch_basal_forcings")
    temporary = (cfg.paths.data_dir / "forcing_download_tmp").resolve()
    data_root = cfg.paths.data_dir.resolve()
    if data_root not in temporary.parents:
        raise RuntimeError("diretório temporário fora de data_dir")
    temporary.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    try:
        for key, item in FILES.items():
            paths[key] = _download(item, temporary, log)

        from netCDF4 import Dataset
        from pyproj import Transformer
        with Dataset(paths["grid"]) as grid:
            lon = np.asarray(grid["lon"][:], dtype=float)
            lat = np.asarray(grid["lat"][:], dtype=float)
        inside = (
            (lon >= cfg.roi.lon_min - args.buffer_deg)
            & (lon <= cfg.roi.lon_max + args.buffer_deg)
            & (lat >= cfg.roi.lat_min - args.buffer_deg)
            & (lat <= cfg.roi.lat_max + args.buffer_deg))
        row_index, col_index = np.where(inside)
        if not len(row_index):
            raise ValueError("ROI não intercepta a grade RACMO")
        rows = slice(max(int(row_index.min()) - 1, 0),
                     min(int(row_index.max()) + 2, lon.shape[0]))
        cols = slice(max(int(col_index.min()) - 1, 0),
                     min(int(col_index.max()) + 2, lon.shape[1]))
        lon_sub, lat_sub = lon[rows, cols], lat[rows, cols]
        mask = (
            (lon_sub >= cfg.roi.lon_min - args.buffer_deg)
            & (lon_sub <= cfg.roi.lon_max + args.buffer_deg)
            & (lat_sub >= cfg.roi.lat_min - args.buffer_deg)
            & (lat_sub <= cfg.roi.lat_max + args.buffer_deg))
        transformer = Transformer.from_crs(4326, 3031, always_xy=True)
        x, y = transformer.transform(lon_sub, lat_sub)
        target = cfg.paths.data_dir / "basal_forcing"
        _write_subset(
            paths["fac"], target / "imau_fdm_fac_ase_2018_2023.nc",
            ("FirnAir", "firnair", "FAC"), args.time_min, mask, rows, cols,
            x, y, lon_sub, lat_sub, output_variable="fac",
            source_doi="10.5281/zenodo.14634258", log=log)
        _write_subset(
            paths["smb"], target / "racmo_smb_ase_2018_2022.nc",
            ("smb", "SMB"), args.time_min, mask, rows, cols,
            x, y, lon_sub, lat_sub, output_variable="smb_monthly",
            source_doi="10.5281/zenodo.7845736", log=log)
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)
        if temporary.exists() and not any(temporary.iterdir()):
            temporary.rmdir()


if __name__ == "__main__":
    main()
