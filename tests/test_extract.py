"""Testes da extração ATL06 (controle de qualidade e conversão de tempo)."""

import numpy as np

from thwaites.io.extract import extract_atl06
from thwaites.io.store import POINT_COLUMNS


def test_extract_valid_count(synthetic_atl06, cfg):
    h5_path, n_expected = synthetic_atl06
    df = extract_atl06(h5_path, cfg)
    assert len(df) == n_expected  # quality_summary=0: 1 por feixe


def test_extract_schema_and_dtypes(synthetic_atl06, cfg):
    h5_path, _ = synthetic_atl06
    df = extract_atl06(h5_path, cfg)
    assert list(df.columns) == list(POINT_COLUMNS.keys())
    assert str(df["h_elv"].dtype) == "float32"
    assert str(df["beam"].dtype) == "int8"
    assert str(df["rgt"].dtype) == "int16"
    assert str(df["segment_id"].dtype) == "int32"


def test_extract_qc_removes_invalids(synthetic_atl06, cfg):
    h5_path, _ = synthetic_atl06
    df = extract_atl06(h5_path, cfg)
    # nenhum fill, absurdo ou NaN sobrevive
    assert df["h_elv"].max() < cfg.qc.fill_value
    assert df["h_elv"].min() > cfg.qc.h_min_valid
    assert not np.isnan(df["lat"]).any()


def test_extract_time_conversion(synthetic_atl06, cfg):
    h5_path, _ = synthetic_atl06
    df = extract_atl06(h5_path, cfg)
    # delta_time foi montado para equivaler a 2021.5
    assert np.allclose(df["t_year"].to_numpy(), 2021.5)


def test_extract_beam_mapping(synthetic_atl06, cfg):
    h5_path, _ = synthetic_atl06
    df = extract_atl06(h5_path, cfg)
    # beams do fixture: gt1l (->1) e gt2r (->4 na ordem da config)
    assert set(df["beam"].unique()) == {1, 4}
    assert set(df["rgt"].unique()) == {1234}
    assert set(df["cycle"].unique()) == {3}


def test_extract_correction_columns(synthetic_atl06, cfg):
    h5_path, _ = synthetic_atl06
    df = extract_atl06(h5_path, cfg)
    # colunas de correção presentes
    for c in ("tide_ocean", "tide_equilibrium", "dac", "geoid"):
        assert c in df.columns
    # um dos dois segmentos sobreviventes tinha fill -> vira NaN
    assert np.isnan(df["tide_ocean"]).sum() == 1
    assert np.allclose(df["tide_ocean"].dropna().to_numpy(), 0.5)
    assert np.allclose(df["dac"].to_numpy(), 0.1)
    assert np.allclose(df["geoid"].to_numpy(), -30.0)


def test_extract_empty_when_no_valid(tmp_path, cfg):
    import h5py
    p = cfg.product
    h5_path = tmp_path / "ATL06_20210715123456_12340312_007_01.h5"
    with h5py.File(h5_path, "w") as f:
        g = f.create_group(f"gt1l/{p.segments_group}")
        g[p.variables["latitude"]]  = np.array([np.nan, np.nan])
        g[p.variables["longitude"]] = np.array([-100.0, -100.1])
        g[p.variables["height"]]    = np.array([3.5e38, 3.5e38])
        g[p.variables["sigma"]]     = np.array([0.1, 0.1])
        g[p.variables["delta_t"]]   = np.array([0.0, 0.0])
        g[p.variables["quality"]]   = np.array([2, 2])
    df = extract_atl06(h5_path, cfg)
    assert len(df) == 0
    # schema completo mesmo vazio (núcleo + colunas de correção)
    assert list(df.columns) == list(POINT_COLUMNS)
