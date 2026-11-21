"""Extração auditável de ICESat-2 ATL06 Release 007.

O contrato v2 preserva no mesmo registro a altura, sua identidade orbital,
segmento, correções e flags de qualidade. Isso elimina o pareamento posicional
com arquivos auxiliares que tornou a ingestão anterior não reprodutível.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.io.store import POINT_COLUMNS
from thwaites.qc.atl06_flags import quality_mask


# output -> (subgrupo de land_ice_segments, variável, dtype numpy)
_ATL06_FIELDS: dict[str, tuple[str, str, str]] = {
    "segment_id": ("", "segment_id", "int32"),
    "x_atc": ("ground_track", "x_atc", "float64"),
    "y_atc": ("ground_track", "y_atc", "float64"),
    # sigma_geo_h fica na raiz de land_ice_segments, NÃO em ground_track (conferido num
    # grânulo real r007 em 2026-11-12; o mapa anterior só passava no dado sintético).
    "sigma_geo_h": ("", "sigma_geo_h", "float32"),
    "sigma_geo_at": ("ground_track", "sigma_geo_at", "float32"),
    "sigma_geo_xt": ("ground_track", "sigma_geo_xt", "float32"),
    "h_robust_sprd": ("fit_statistics", "h_robust_sprd", "float32"),
    "n_fit_photons": ("fit_statistics", "n_fit_photons", "int32"),
    "dh_fit_dx": ("fit_statistics", "dh_fit_dx", "float32"),
    "h_expected_rms": ("fit_statistics", "h_expected_rms", "float32"),
    "snr_significance": ("fit_statistics", "snr_significance", "float32"),
    "msw_flag": ("geophysical", "msw_flag", "int8"),
    "cloud_flg_asr": ("geophysical", "cloud_flg_asr", "int8"),
    "cloud_flg_atm": ("geophysical", "cloud_flg_atm", "int8"),
    "bsnow_conf": ("geophysical", "bsnow_conf", "int8"),
    "solar_elevation": ("geophysical", "solar_elevation", "float32"),
    "dem_h": ("dem", "dem_h", "float32"),
}


def _beam_index_map(beams: list[str]) -> dict[str, int]:
    return {b: i + 1 for i, b in enumerate(beams)}


def _granule_identity(path: Path) -> tuple[str, int, int, int]:
    """Extrai granule_id, RGT, ciclo e região do nome oficial ATL06."""
    stem = path.stem
    match = re.fullmatch(r"ATL06_\d{14}_(\d{4})(\d{2})(\d{2})_\d{3}_\d{2}", stem)
    if match is None:
        raise ValueError(
            f"nome ATL06 fora do padrão oficial (identidade orbital ausente): {path.name}")
    rgt, cycle, region = (int(value) for value in match.groups())
    return stem, rgt, cycle, region


def _required(seg: h5py.Group, group_name: str, var_name: str) -> h5py.Dataset:
    group = seg if not group_name else seg.get(group_name)
    if group is None or var_name not in group:
        path = f"{seg.name}/{group_name}/{var_name}".replace("//", "/")
        raise KeyError(f"campo obrigatório do contrato ATL06 R007 ausente: {path}")
    return group[var_name]


def _empty_frame() -> pd.DataFrame:
    dtypes = {
        "string": "string", "float64": "float64", "float32": "float32",
        "int8": "int8", "int16": "int16", "int32": "int32",
    }
    return pd.DataFrame({
        name: pd.Series(dtype=dtypes[dtype])
        for name, dtype in POINT_COLUMNS.items()
    })


def extract_atl06(h5_path: str | Path, cfg: Config) -> pd.DataFrame:
    """Extrai um grânulo ATL06 R007 e aplica o QC configurado fail-closed.

    Se um campo obrigatório estiver ausente, a extração falha; o chamador apaga
    o HDF5 em ``finally`` e não publica Parquet parcial. Correções geofísicas
    fill são preservadas como NaN para decisão explícita na etapa de correção.
    """
    h5_path = Path(h5_path)
    granule_id, rgt, cycle, region = _granule_identity(h5_path)
    p = cfg.product
    v = p.variables
    beam_map = _beam_index_map(p.beams)
    fill_thr = cfg.corrections.fill_threshold
    parts: dict[str, list[np.ndarray]] = {name: [] for name in POINT_COLUMNS}
    missing_corrections: set[str] = set(p.correction_vars)

    with h5py.File(h5_path, "r") as handle:
        for beam in p.beams:
            if beam not in handle or p.segments_group not in handle[beam]:
                continue
            seg = handle[f"{beam}/{p.segments_group}"]
            if not all(name in seg for name in v.values()):
                raise KeyError(f"variável nuclear ATL06 ausente em {seg.name}")

            lat = np.asarray(seg[v["latitude"]][:])
            lon = np.asarray(seg[v["longitude"]][:])
            height = np.asarray(seg[v["height"]][:])
            sigma = np.asarray(seg[v["sigma"]][:])
            delta_t = np.asarray(seg[v["delta_t"]][:])
            quality = np.asarray(seg[v["quality"]][:])
            valid = (
                np.isfinite(lat) & np.isfinite(lon) & np.isfinite(height)
                & np.isfinite(delta_t) & (height < cfg.qc.fill_value)
                & (height > cfg.qc.h_min_valid)
            )
            if not valid.any():
                continue
            n = int(valid.sum())

            parts["granule_id"].append(np.full(n, granule_id, dtype=object))
            parts["rgt"].append(np.full(n, rgt, dtype=np.int16))
            parts["cycle"].append(np.full(n, cycle, dtype=np.int8))
            parts["region"].append(np.full(n, region, dtype=np.int8))
            parts["lon"].append(lon[valid].astype(np.float64))
            parts["lat"].append(lat[valid].astype(np.float64))
            parts["h_elv"].append(height[valid].astype(np.float32))
            parts["s_elv"].append(sigma[valid].astype(np.float32))
            parts["delta_time"].append(delta_t[valid].astype(np.float64))
            parts["t_year"].append(
                (p.atlas_epoch_year + delta_t[valid] / p.seconds_per_year).astype(np.float64))
            parts["beam"].append(np.full(n, beam_map[beam], dtype=np.int8))
            parts["atl06_quality_summary"].append(quality[valid].astype(np.int8))

            for out_name, (group_name, var_name, dtype) in _ATL06_FIELDS.items():
                if out_name == "atl06_quality_summary":
                    continue
                array = np.asarray(_required(seg, group_name, var_name)[:])[valid]
                if dtype.startswith("float"):
                    array = np.where(np.abs(array) > fill_thr, np.nan, array)
                parts[out_name].append(array.astype(dtype))

            for out_name, h5name in p.correction_vars.items():
                if "/" in h5name:
                    group_name, var_name = h5name.split("/", 1)
                else:
                    group_name, var_name = p.geophysical_group, h5name
                group = seg.get(group_name)
                if group is not None and var_name in group:
                    array = np.asarray(group[var_name][:], dtype=np.float64)[valid]
                    array = np.where(np.abs(array) > fill_thr, np.nan, array)
                    missing_corrections.discard(out_name)
                else:
                    array = np.full(n, np.nan)
                parts[out_name].append(array.astype(np.float32))

    if missing_corrections:
        warnings.warn(
            "correções ausentes no grânulo e marcadas como NaN: "
            f"{sorted(missing_corrections)}", RuntimeWarning, stacklevel=2)
    if not parts["lat"]:
        return _empty_frame()

    frame = pd.DataFrame({name: np.concatenate(values) for name, values in parts.items()})
    if cfg.atl06_qc.enabled:
        keep, reasons = quality_mask(
            frame, cfg, h_li=frame["h_elv"].to_numpy(dtype=float))
        qc_attrs = {
            "qc_n_in": int(len(frame)), "qc_n_out": int(keep.sum()),
            "qc_by_reason": reasons,
        }
        frame = frame.loc[keep].reset_index(drop=True)
        frame.attrs.update(qc_attrs)

    frame = frame.loc[:, list(POINT_COLUMNS)]
    for name, dtype in POINT_COLUMNS.items():
        frame[name] = frame[name].astype(dtype)
    return frame
