"""Leitura do recorte de velocidade do gelo (fetch_velocity.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thwaites.config import Config


def load_velocity(cfg: Config, path: str | Path | None = None):
    """
    Carrega o recorte de velocidade (fetch_velocity.py) como (x, y, vx, vy).

    Retorna arrays 1D x, y (EPSG:3031) e 2D vx, vy (m/ano), com nodata -> NaN.
    """
    import xarray as xr

    p = Path(path) if path else (cfg.paths.data_dir / cfg.velocity.path)
    if not p.exists():
        raise FileNotFoundError(
            f"Velocidade não encontrada: {p}\nRode pipelines/fetch_velocity.py.")
    ds = xr.open_dataset(p)
    vx = np.asarray(ds[cfg.velocity.vx_var].values, dtype=np.float64)
    vy = np.asarray(ds[cfg.velocity.vy_var].values, dtype=np.float64)
    x = np.asarray(ds["x"].values, dtype=np.float64)
    y = np.asarray(ds["y"].values, dtype=np.float64)
    # normaliza para y crescente (facilita o gradiente)
    if y[0] > y[-1]:
        y = y[::-1]
        vx = vx[::-1, :]
        vy = vy[::-1, :]
    for a in (vx, vy):
        a[np.abs(a) > 1e9] = np.nan       # fill values típicos
    ds.close()
    return x, y, vx, vy
