"""Testes de armazenamento Parquet: roundtrip e consolidação."""

import numpy as np
import pandas as pd
import pytest

from thwaites.io.store import (
    save_points_parquet, read_points_parquet, consolidate_parquets, POINT_COLUMNS,
)


def _sample_df(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "granule_id": ["ATL06_20210715123456_12340312_007_01"] * n,
        "rgt": np.full(n, 1234, dtype="int16"),
        "cycle": np.full(n, 3, dtype="int8"),
        "region": np.full(n, 12, dtype="int8"),
        "lon": rng.uniform(-114, -70, n).astype("float64"),
        "lat": rng.uniform(-80, -69, n).astype("float64"),
        "h_elv": rng.uniform(0, 1000, n).astype("float32"),
        "s_elv": rng.uniform(0, 1, n).astype("float32"),
        "delta_time": rng.uniform(31_536_000, 220_000_000, n).astype("float64"),
        "t_year": rng.uniform(2019, 2025, n).astype("float64"),
        "beam": rng.integers(1, 7, n).astype("int8"),
        "segment_id": np.arange(n, dtype="int32"),
        "x_atc": np.arange(n, dtype="float64") * 40.0,
        "y_atc": np.zeros(n, dtype="float64"),
        "atl06_quality_summary": np.zeros(n, dtype="int8"),
        "h_robust_sprd": np.full(n, 0.2, dtype="float32"),
        "n_fit_photons": np.full(n, 20, dtype="int32"),
        "dh_fit_dx": np.zeros(n, dtype="float32"),
        "h_expected_rms": np.full(n, 0.1, dtype="float32"),
        "snr_significance": np.zeros(n, dtype="float32"),
        "msw_flag": np.zeros(n, dtype="int8"),
        "cloud_flg_asr": np.zeros(n, dtype="int8"),
        "cloud_flg_atm": np.zeros(n, dtype="int8"),
        "bsnow_conf": np.full(n, -1, dtype="int8"),
        "solar_elevation": np.full(n, -10.0, dtype="float32"),
        "dem_h": rng.uniform(0, 1000, n).astype("float32"),
        "sigma_geo_h": np.full(n, 0.03, dtype="float32"),
        "sigma_geo_at": np.full(n, 3.0, dtype="float32"),
        "sigma_geo_xt": np.full(n, 3.0, dtype="float32"),
        "tide_earth": np.zeros(n, dtype="float32"),
        "tide_load": np.zeros(n, dtype="float32"),
        "tide_ocean": rng.uniform(-1, 1, n).astype("float32"),
        "tide_pole": np.zeros(n, dtype="float32"),
        "tide_equilibrium": rng.uniform(-0.05, 0.05, n).astype("float32"),
        "dac": rng.uniform(-0.5, 0.5, n).astype("float32"),
        "neutat_delay_total": np.zeros(n, dtype="float32"),
        "geoid": rng.uniform(-40, -20, n).astype("float32"),
    })


def test_roundtrip(tmp_path):
    df = _sample_df()
    path = save_points_parquet(df, tmp_path / "g1.parquet")
    back = read_points_parquet(path)
    pd.testing.assert_frame_equal(df, back)


def test_save_rejects_bad_schema(tmp_path):
    df = pd.DataFrame({"lon": [1.0], "lat": [2.0]})  # faltam colunas
    with pytest.raises(ValueError):
        save_points_parquet(df, tmp_path / "bad.parquet")


def test_consolidate(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    save_points_parquet(_sample_df(5, seed=1), d / "g1.parquet")
    save_points_parquet(_sample_df(7, seed=2), d / "g2.parquet")
    out, n = consolidate_parquets(d, tmp_path / "interim" / "merged.parquet")
    assert n == 12
    merged = read_points_parquet(out)
    assert len(merged) == 12
    assert list(merged.columns) == list(POINT_COLUMNS.keys())


def test_consolidate_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        consolidate_parquets(d, tmp_path / "out.parquet")


def _df_at(lon, lat, n=5):
    """DataFrame com todos os pontos numa dada (lon, lat)."""
    base = _sample_df(n)
    base["lon"] = float(lon)
    base["lat"] = float(lat)
    return base


def test_consolidate_roi_filters_and_logs(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    # arquivo A dentro da ROI, arquivo B fora
    save_points_parquet(_df_at(-107.0, -75.0, n=5), d / "granuleA.parquet")
    save_points_parquet(_df_at(-90.0, -72.0, n=5), d / "granuleB.parquet")

    roi = (-110.0, -76.0, -104.0, -74.0)   # (lon_min, lat_min, lon_max, lat_max)
    log_md = tmp_path / "recorte.md"
    out, n = consolidate_parquets(d, tmp_path / "merged.parquet",
                                  roi=roi, exclusion_log=log_md)

    # só os 5 pontos de A sobraram
    assert n == 5
    merged = read_points_parquet(out)
    assert len(merged) == 5
    assert (merged["lon"] == -107.0).all()

    # log criado e lista o arquivo B como desconsiderado (e não o A)
    assert log_md.exists()
    text = log_md.read_text(encoding="utf-8")
    assert "granuleB.parquet" in text
    assert "granuleA.parquet" not in text.split("## Arquivos desconsiderados")[1]


def test_consolidate_qc_fails_closed_and_preserves_previous_output(tmp_path, cfg):
    d = tmp_path / "processed"
    q = tmp_path / "qc_flags"
    d.mkdir()
    q.mkdir()
    save_points_parquet(_df_at(-107.0, -75.0, n=5), d / "granuleA.parquet")

    out = tmp_path / "merged.parquet"
    previous = _sample_df(2, seed=99)
    save_points_parquet(previous, out)

    cfg.atl06_qc.enabled = True
    cfg.atl06_qc.require_complete_flags = True
    with pytest.raises(FileNotFoundError, match="flags ATL06 ausentes"):
        consolidate_parquets(
            d, out, roi=(-110.0, -76.0, -104.0, -74.0),
            qc_dir=q, cfg=cfg, qc_stats={})

    # A escrita transacional não troca a saída válida por um arquivo parcial.
    pd.testing.assert_frame_equal(previous, read_points_parquet(out))
    assert not (tmp_path / "merged.parquet.partial").exists()
