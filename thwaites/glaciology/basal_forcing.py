"""Amostragem co-temporal de FAC e SMB sem extrapolação.

Os produtos nativos de 27 km são mantidos como nuvens de centros de célula em
EPSG:3031. A interpolação espacial usa quatro vizinhos com peso inverso da
distância; a dispersão entre vizinhos quantifica representatividade subgrade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PointForcingCube:
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    units: str
    source: str

    @classmethod
    def from_native_subset(cls, path: Path, variable: str):
        from netCDF4 import Dataset

        with Dataset(path) as dataset:
            time = np.asarray(dataset["time"][:], dtype=float)
            x = np.asarray(dataset["x"][:], dtype=float)
            y = np.asarray(dataset["y"][:], dtype=float)
            values = np.ma.filled(
                np.ma.asarray(dataset[variable][:], dtype=float), np.nan)
            units = str(getattr(dataset[variable], "units", ""))
            source = str(getattr(dataset, "source_doi", path.name))
        return cls(time, x, y, values, units, source)

    @classmethod
    def from_regular_grid(cls, path: Path, variable: str):
        from netCDF4 import Dataset

        with Dataset(path) as dataset:
            time = np.asarray(dataset["time"][:], dtype=float)
            x_axis = np.asarray(dataset["x"][:], dtype=float)
            y_axis = np.asarray(dataset["y"][:], dtype=float)
            x, y = np.meshgrid(x_axis, y_axis)
            values = np.ma.filled(
                np.ma.asarray(dataset[variable][:], dtype=float), np.nan)
            units = str(getattr(dataset[variable], "units", ""))
        return cls(time, x.ravel(), y.ravel(),
                   values.reshape(len(time), -1), units, path.name)

    def neighbor_series(self, px, py, k: int = 4):
        from scipy.spatial import cKDTree

        query = np.c_[np.asarray(px, float), np.asarray(py, float)]
        distance, index = cKDTree(np.c_[self.x, self.y]).query(
            query, k=min(k, len(self.x)))
        if distance.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        inverse = 1.0 / np.maximum(distance, 100.0)
        weight = inverse / inverse.sum(axis=1, keepdims=True)
        # (tempo, parcela, vizinho)
        return self.values[:, index], weight, distance


def sample_at_times(cube: PointForcingCube, px, py, times,
                    k: int = 4) -> dict[str, np.ndarray]:
    """Interpolação linear no tempo e IDW espacial em épocas individuais."""
    from scipy.spatial import cKDTree

    px = np.asarray(px, float)
    py = np.asarray(py, float)
    query_time = np.asarray(times, float)
    distance, index = cKDTree(np.c_[cube.x, cube.y]).query(
        np.c_[px, py], k=min(k, len(cube.x)))
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    inverse = 1.0 / np.maximum(distance, 100.0)
    weight = inverse / inverse.sum(axis=1, keepdims=True)
    upper = np.searchsorted(cube.time, query_time, side="right")
    lower = upper - 1
    # Uma consulta exatamente igual à última época é suportada. Sem este caso,
    # searchsorted(..., side="right") a classificaria como extrapolação.
    exact_last = np.isclose(query_time, cube.time[-1], rtol=0.0, atol=1e-10)
    supported = ((lower >= 0) & (upper < len(cube.time))) | exact_last
    lower = np.clip(lower, 0, len(cube.time) - 1)
    upper = np.clip(upper, 0, len(cube.time) - 1)
    lower[exact_last] = len(cube.time) - 1
    upper[exact_last] = len(cube.time) - 1
    dt = cube.time[upper] - cube.time[lower]
    fraction = np.divide(
        query_time - cube.time[lower], dt,
        out=np.zeros_like(query_time), where=dt > 0)
    first = cube.values[lower[:, None], index]
    second = cube.values[upper[:, None], index]
    neighbor = first + fraction[:, None] * (second - first)
    finite = np.isfinite(neighbor)
    normalized = np.where(finite, weight, 0.0)
    denominator = normalized.sum(axis=1)
    value = np.divide(
        np.nansum(neighbor * normalized, axis=1), denominator,
        out=np.full(len(px), np.nan), where=denominator > 0)
    # Dispersão entre vizinhos (ddof=1 sobre os finitos), vetorizada: a versão
    # por compreensão custava minutos em milhões de parcelas.
    n_finite = finite.sum(axis=1)
    with np.errstate(invalid="ignore"):
        media = np.nanmean(np.where(finite, neighbor, np.nan), axis=1)
        soma_quad = np.nansum(
            np.where(finite, (neighbor - media[:, None]) ** 2, 0.0), axis=1)
    sigma_rep = np.where(
        n_finite > 1, np.sqrt(soma_quad / np.maximum(n_finite - 1, 1)), np.nan)
    value[~supported] = np.nan
    sigma_rep[~supported] = np.nan
    return {"value": value, "sigma_rep": sigma_rep,
            "nearest_distance_m": distance[:, 0], "supported": supported}


def _slope(time: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(time) & np.isfinite(values)
    if finite.sum() < 4:
        return float("nan")
    t = time[finite]
    v = values[finite]
    centered = t - t.mean()
    denominator = float(np.sum(centered ** 2))
    return (float(np.sum(centered * (v - v.mean())) / denominator)
            if denominator > 0 else float("nan"))


def _fac_one(cube: PointForcingCube, series: np.ndarray,
             weights: np.ndarray, start: float, end: float) -> dict:
    half_step = 0.5 * float(np.nanmedian(np.diff(cube.time)))
    if (start < cube.time.min() - half_step
            or end > cube.time.max() + half_step or end <= start):
        return {key: np.nan for key in
                ("rate", "sigma_rep", "sigma_time", "coverage")}
    select = (cube.time >= start) & (cube.time <= end)
    if select.sum() < 20:
        return {key: np.nan for key in
                ("rate", "sigma_rep", "sigma_time", "coverage")}
    t = cube.time[select]
    v = series[select]
    slopes = np.asarray([_slope(t, v[:, neighbor])
                         for neighbor in range(v.shape[1])])
    central_series = np.nansum(v * weights[None, :], axis=1)
    central = _slope(t, central_series)
    valid_slopes = slopes[np.isfinite(slopes)]
    sigma_rep = (float(np.std(valid_slopes, ddof=1))
                 if len(valid_slopes) > 1 else np.nan)

    years = np.floor(t).astype(int)
    leave_year = np.asarray([
        _slope(t[years != year], central_series[years != year])
        for year in np.unique(years)
    ])
    leave_year = leave_year[np.isfinite(leave_year)]
    if len(leave_year) >= 3:
        mean = leave_year.mean()
        sigma_time = float(np.sqrt(
            (len(leave_year) - 1) / len(leave_year)
            * np.sum((leave_year - mean) ** 2)))
    else:
        sigma_time = np.nan
    return {
        "rate": central,
        "sigma_rep": sigma_rep,
        "sigma_time": sigma_time,
        "coverage": float((t.max() - t.min()) / (end - start)),
    }


def fac_rates(cube: PointForcingCube, px, py, starts, ends) -> dict[str, np.ndarray]:
    series, weights, distance = cube.neighbor_series(px, py)
    rows = [_fac_one(cube, series[:, i, :], weights[i], float(start), float(end))
            for i, (start, end) in enumerate(zip(starts, ends))]
    result = {key: np.asarray([row[key] for row in rows], dtype=float)
              for key in rows[0]}
    result["nearest_distance_m"] = distance[:, 0]
    return result


def _mean_rate_one(cube: PointForcingCube, series: np.ndarray,
                   weights: np.ndarray, start: float, end: float,
                   *, monthly_accumulation: bool,
                   ice_density: float) -> dict:
    half_step = 0.5 * float(np.nanmedian(np.diff(cube.time)))
    if (start < cube.time.min() - half_step
            or end > cube.time.max() + half_step or end <= start):
        return {key: np.nan for key in ("rate", "sigma_rep", "coverage")}
    select = (cube.time >= start) & (cube.time <= end)
    minimum = 12 if monthly_accumulation else 20
    if select.sum() < minimum:
        return {key: np.nan for key in ("rate", "sigma_rep", "coverage")}
    values = series[select]
    if monthly_accumulation:
        # RACMO armazena o acumulado de cada mês em kg m-2.
        rates = np.nansum(values, axis=0) / ice_density / (select.sum() / 12.0)
        covered = select.sum() / 12.0
    else:
        # GSFC-FDM armazena uma taxa instantânea em m gelo ano-1.
        rates = np.nanmean(values, axis=0)
        selected_time = cube.time[select]
        covered = float(selected_time.max() - selected_time.min())
    finite = np.isfinite(rates)
    central = (float(np.sum(rates[finite] * weights[finite])
                     / np.sum(weights[finite])) if finite.any() else np.nan)
    sigma_rep = (float(np.std(rates[finite], ddof=1))
                 if finite.sum() > 1 else np.nan)
    return {
        "rate": central, "sigma_rep": sigma_rep,
        "coverage": float(covered / (end - start)),
    }


def mean_rates(cube: PointForcingCube, px, py, starts, ends, *,
               monthly_accumulation: bool, ice_density: float) -> dict[str, np.ndarray]:
    series, weights, distance = cube.neighbor_series(px, py)
    rows = [_mean_rate_one(
        cube, series[:, i, :], weights[i], float(start), float(end),
        monthly_accumulation=monthly_accumulation, ice_density=ice_density)
        for i, (start, end) in enumerate(zip(starts, ends))]
    result = {key: np.asarray([row[key] for row in rows], dtype=float)
              for key in rows[0]}
    result["nearest_distance_m"] = distance[:, 0]
    return result


def _moving_neighbors(cube: PointForcingCube, x0: float, y0: float,
                      x1: float, y1: float, start: float, end: float,
                      k: int = 4):
    from scipy.spatial import cKDTree

    select = (cube.time >= start) & (cube.time <= end)
    t = cube.time[select]
    if not len(t):
        return t, np.empty((0, k)), np.empty((0, k)), np.empty((0, k))
    fraction = np.clip((t - start) / max(end - start, 1e-12), 0.0, 1.0)
    x = x0 + fraction * (x1 - x0)
    y = y0 + fraction * (y1 - y0)
    distance, index = cKDTree(np.c_[cube.x, cube.y]).query(
        np.c_[x, y], k=min(k, len(cube.x)))
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    inverse = 1.0 / np.maximum(distance, 100.0)
    weight = inverse / inverse.sum(axis=1, keepdims=True)
    time_index = np.flatnonzero(select)
    values = cube.values[time_index[:, None], index]
    return t, values, weight, distance


def moving_fac_rates(cube: PointForcingCube, x0, y0, x1, y1,
                     starts, ends) -> dict[str, np.ndarray]:
    rows = []
    half_step = 0.5 * float(np.nanmedian(np.diff(cube.time)))
    for values in zip(x0, y0, x1, y1, starts, ends):
        px0, py0, px1, py1, start, end = map(float, values)
        if (start < cube.time.min() - half_step
                or end > cube.time.max() + half_step or end <= start):
            rows.append({"rate": np.nan, "sigma_rep": np.nan,
                         "sigma_time": np.nan, "coverage": np.nan,
                         "nearest_distance_m": np.nan})
            continue
        t, neighbors, weights, distance = _moving_neighbors(
            cube, px0, py0, px1, py1, start, end)
        if len(t) < 20:
            rows.append({"rate": np.nan, "sigma_rep": np.nan,
                         "sigma_time": np.nan, "coverage": np.nan,
                         "nearest_distance_m": np.nan})
            continue
        central_series = np.nansum(neighbors * weights, axis=1)
        rate = _slope(t, central_series)
        neighbor_slopes = np.asarray([
            _slope(t, neighbors[:, column])
            for column in range(neighbors.shape[1])])
        finite = neighbor_slopes[np.isfinite(neighbor_slopes)]
        sigma_rep = (float(np.std(finite, ddof=1))
                     if len(finite) > 1 else np.nan)
        years = np.floor(t).astype(int)
        leave_year = np.asarray([
            _slope(t[years != year], central_series[years != year])
            for year in np.unique(years)])
        leave_year = leave_year[np.isfinite(leave_year)]
        if len(leave_year) >= 3:
            mean = leave_year.mean()
            sigma_time = float(np.sqrt(
                (len(leave_year) - 1) / len(leave_year)
                * np.sum((leave_year - mean) ** 2)))
        else:
            sigma_time = np.nan
        rows.append({
            "rate": rate, "sigma_rep": sigma_rep,
            "sigma_time": sigma_time,
            "coverage": float((t.max() - t.min()) / (end - start)),
            "nearest_distance_m": float(np.nanmax(distance[:, 0])),
        })
    return {key: np.asarray([row[key] for row in rows], dtype=float)
            for key in rows[0]}


def moving_mean_rates(cube: PointForcingCube, x0, y0, x1, y1,
                      starts, ends, *, monthly_accumulation: bool,
                      ice_density: float) -> dict[str, np.ndarray]:
    rows = []
    half_step = 0.5 * float(np.nanmedian(np.diff(cube.time)))
    for values in zip(x0, y0, x1, y1, starts, ends):
        px0, py0, px1, py1, start, end = map(float, values)
        if (start < cube.time.min() - half_step
                or end > cube.time.max() + half_step or end <= start):
            rows.append({"rate": np.nan, "sigma_rep": np.nan,
                         "coverage": np.nan, "nearest_distance_m": np.nan})
            continue
        if monthly_accumulation:
            # Os valores RACMO são acumulados mensais associados ao centro do
            # mês. Integram-se frações dos meses de borda, em vez de exigir um
            # ano completo ou selecionar apenas centros dentro do intervalo.
            midpoint = 0.5 * (cube.time[:-1] + cube.time[1:])
            bounds = np.r_[cube.time[0] - (midpoint[0] - cube.time[0]),
                           midpoint,
                           cube.time[-1] + (cube.time[-1] - midpoint[-1])]
            overlap = np.maximum(
                0.0, np.minimum(end, bounds[1:])
                - np.maximum(start, bounds[:-1]))
            index_time = np.flatnonzero(overlap > 0)
            if not len(index_time):
                rows.append({"rate": np.nan, "sigma_rep": np.nan,
                             "coverage": np.nan,
                             "nearest_distance_m": np.nan})
                continue
            from scipy.spatial import cKDTree
            t = cube.time[index_time]
            fraction_path = np.clip(
                (t - start) / max(end - start, 1e-12), 0.0, 1.0)
            x = px0 + fraction_path * (px1 - px0)
            y = py0 + fraction_path * (py1 - py0)
            distance, spatial_index = cKDTree(
                np.c_[cube.x, cube.y]).query(
                    np.c_[x, y], k=min(4, len(cube.x)))
            if distance.ndim == 1:
                distance = distance[:, None]
                spatial_index = spatial_index[:, None]
            inverse = 1.0 / np.maximum(distance, 100.0)
            weights = inverse / inverse.sum(axis=1, keepdims=True)
            neighbors = cube.values[index_time[:, None], spatial_index]
            central = np.nansum(neighbors * weights, axis=1)
            month_duration = bounds[1:] - bounds[:-1]
            share = overlap[index_time] / month_duration[index_time]
            # kg m-2 acumulados no intervalo / rho_i / duração em anos.
            rate = float(np.nansum(central * share) / ice_density
                         / (end - start))
            neighbor_rates = np.nansum(
                neighbors * share[:, None], axis=0) / ice_density / (end - start)
            covered = float(np.sum(overlap[index_time]))
        else:
            t, neighbors, weights, distance = _moving_neighbors(
                cube, px0, py0, px1, py1, start, end)
            if len(t) < 2:
                rows.append({"rate": np.nan, "sigma_rep": np.nan,
                             "coverage": np.nan,
                             "nearest_distance_m": np.nan})
                continue
            central = np.nansum(neighbors * weights, axis=1)
            rate = float(np.nanmean(central))
            neighbor_rates = np.nanmean(neighbors, axis=0)
            covered = float(t.max() - t.min())
        if not np.isfinite(rate):
            rows.append({"rate": np.nan, "sigma_rep": np.nan,
                         "coverage": np.nan, "nearest_distance_m": np.nan})
            continue
        finite = neighbor_rates[np.isfinite(neighbor_rates)]
        rows.append({
            "rate": rate,
            "sigma_rep": (float(np.std(finite, ddof=1))
                          if len(finite) > 1 else np.nan),
            "coverage": float(covered / (end - start)),
            "nearest_distance_m": float(np.nanmax(distance[:, 0])),
        })
    return {key: np.asarray([row[key] for row in rows], dtype=float)
            for key in rows[0]}
