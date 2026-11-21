"""Divergência de velocidade e sua incerteza em grades regulares."""

from __future__ import annotations

import numpy as np


def smooth_nan(array, sigma_px):
    from scipy.ndimage import gaussian_filter

    if sigma_px <= 0:
        return np.asarray(array, dtype=float)
    array = np.asarray(array, dtype=float)
    valid = np.isfinite(array)
    numerator = gaussian_filter(np.where(valid, array, 0.0), sigma_px)
    denominator = gaussian_filter(valid.astype(float), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 1e-6, numerator / denominator, np.nan)


def divergence_field(vx, vy, dx, dy, smooth_px):
    """Retorna ``du/dx+dv/dy`` em ano-1 para velocidade em m ano-1."""
    vxs = smooth_nan(vx, smooth_px)
    vys = smooth_nan(vy, smooth_px)
    return np.gradient(vxs, dx, axis=1) + np.gradient(vys, dy, axis=0)


def divergence_random_sigma(vx_error, vy_error, dx, dy, smooth_px):
    """Propaga erros independentes de velocidade por diferenças centrais."""
    ex = smooth_nan(np.abs(vx_error), smooth_px)
    ey = smooth_nan(np.abs(vy_error), smooth_px)
    sx = np.full_like(ex, np.nan, dtype=float)
    sy = np.full_like(ey, np.nan, dtype=float)
    sx[:, 1:-1] = np.hypot(ex[:, 2:], ex[:, :-2]) / (2 * abs(dx))
    sy[1:-1, :] = np.hypot(ey[2:, :], ey[:-2, :]) / (2 * abs(dy))
    return np.hypot(sx, sy)


def sample_nearest(field, x_axis, y_axis, px, py):
    px, py = np.asarray(px), np.asarray(py)
    j = np.rint((px - x_axis[0]) / (x_axis[1] - x_axis[0])).astype(np.int64)
    i = np.rint((py - y_axis[0]) / (y_axis[1] - y_axis[0])).astype(np.int64)
    valid = (j >= 0) & (j < len(x_axis)) & (i >= 0) & (i < len(y_axis))
    out = np.full(len(px), np.nan)
    out[valid] = field[i[valid], j[valid]]
    return out


def sample_time(cube, times, x_axis, y_axis, px, py, query_time):
    """Interpola linearmente uma grade anual, sem extrapolação temporal."""
    query_time = np.asarray(query_time, dtype=float)
    out = np.full(len(query_time), np.nan)
    for k in range(len(times) - 1):
        select = ((query_time >= times[k]) & (query_time <= times[k + 1]))
        if not select.any():
            continue
        weight = ((query_time[select] - times[k])
                  / max(times[k + 1] - times[k], 1e-12))
        first = sample_nearest(
            cube[k], x_axis, y_axis, np.asarray(px)[select], np.asarray(py)[select])
        second = sample_nearest(
            cube[k + 1], x_axis, y_axis,
            np.asarray(px)[select], np.asarray(py)[select])
        out[select] = (1.0 - weight) * first + weight * second
    exact_last = np.isclose(query_time, times[-1], rtol=0.0, atol=1e-10)
    if exact_last.any():
        out[exact_last] = sample_nearest(
            cube[-1], x_axis, y_axis,
            np.asarray(px)[exact_last], np.asarray(py)[exact_last])
    return out
