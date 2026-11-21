"""Testes do contrato fail-closed para os flags nativos do ATL06."""

import numpy as np
import pandas as pd
import pytest

from thwaites.qc.atl06_flags import quality_mask


def _active_flags():
    return pd.DataFrame({
        "atl06_quality_summary": [0.0, 0.0],
        "h_robust_sprd": [0.2, 0.2],
        "msw_flag": [0.0, 0.0],
        "dem_h": [100.0, 100.0],
    })


def test_active_missing_flag_raises(cfg):
    qc = _active_flags().drop(columns="h_robust_sprd")
    with pytest.raises(ValueError, match="h_robust_sprd"):
        quality_mask(qc, cfg, h_li=np.array([100.0, 100.0]))


def test_nonfinite_active_flag_is_rejected_and_reported(cfg):
    qc = _active_flags()
    qc.loc[1, "h_robust_sprd"] = np.nan
    keep, reasons = quality_mask(qc, cfg, h_li=np.array([100.0, 100.0]))
    assert keep.tolist() == [True, False]
    assert reasons["h_robust_sprd indisponível"] == 1

