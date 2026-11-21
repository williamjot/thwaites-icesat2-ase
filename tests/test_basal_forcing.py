import numpy as np

from thwaites.glaciology.basal_forcing import (
    PointForcingCube, fac_rates, mean_rates, moving_fac_rates,
    moving_mean_rates, sample_at_times)


def _cube(values, time):
    return PointForcingCube(
        np.asarray(time, float),
        np.array([0.0, 1000.0, 0.0, 1000.0]),
        np.array([0.0, 0.0, 1000.0, 1000.0]),
        np.asarray(values, float), "", "synthetic")


def test_fac_rate_and_fail_closed_coverage():
    time = np.linspace(2019.0, 2022.9, 144)
    offsets = np.array([0.0, 0.01, -0.01, 0.02])
    values = 10.0 + 0.2 * (time[:, None] - 2019.0) + offsets
    result = fac_rates(_cube(values, time), [500, 500], [500, 500],
                       [2019.1, 2018.9], [2022.8, 2022.0])
    assert np.isclose(result["rate"][0], 0.2, atol=1e-10)
    assert np.isnan(result["rate"][1])


def test_monthly_smb_accumulation_converts_to_ice_rate():
    time = 2019.0 + (np.arange(48) + 0.5) / 12.0
    # 917 kg m-2 por ano, distribuídos em 12 meses -> 1 m gelo ano-1.
    values = np.full((48, 4), 917.0 / 12.0)
    result = mean_rates(
        _cube(values, time), [500], [500], [2019.0], [2022.999],
        monthly_accumulation=True, ice_density=917.0)
    assert np.isclose(result["rate"][0], 1.0)


def test_moving_samplers_follow_linear_field():
    time = np.linspace(2019.0, 2022.9, 144)
    values = 5.0 + 0.3 * (time[:, None] - 2019.0)
    values = np.repeat(values, 4, axis=1)
    fac = moving_fac_rates(
        _cube(values, time), [0], [0], [1000], [1000], [2019.1], [2022.8])
    assert np.isclose(fac["rate"][0], 0.3, atol=1e-10)

    months = 2019.0 + (np.arange(48) + 0.5) / 12.0
    monthly = _cube(np.full((48, 4), 917.0 / 12.0), months)
    smb = moving_mean_rates(
        monthly, [0], [0], [1000], [1000], [2019.0], [2022.999],
        monthly_accumulation=True, ice_density=917.0)
    assert np.isclose(smb["rate"][0], 1.0)


def test_point_sampler_interpolates_time_and_space():
    cube = _cube(np.array([[0.0, 2.0, 2.0, 4.0],
                           [10.0, 12.0, 12.0, 14.0]]), [2020.0, 2021.0])
    result = sample_at_times(cube, [500.0], [500.0], [2020.5])
    assert np.isclose(result["value"][0], 7.0)


def test_point_sampler_accepts_exact_last_epoch_without_extrapolating():
    cube = _cube(np.array([[1.0] * 4, [3.0] * 4]), [2020.0, 2021.0])
    result = sample_at_times(cube, [500.0, 500.0], [500.0, 500.0],
                             [2021.0, 2021.01])
    assert result["supported"][0]
    assert np.isclose(result["value"][0], 3.0)
    assert not result["supported"][1]
    assert np.isnan(result["value"][1])


def test_quarterly_smb_integrates_partial_months():
    months = 2019.0 + (np.arange(24) + 0.5) / 12.0
    cube = _cube(np.full((24, 4), 917.0 / 12.0), months)
    result = moving_mean_rates(
        cube, [0], [0], [250], [250], [2019.10], [2019.35],
        monthly_accumulation=True, ice_density=917.0)
    assert np.isclose(result["rate"][0], 1.0, rtol=1e-3)
    assert np.isclose(result["coverage"][0], 1.0, rtol=1e-10)
