"""
Fixtures compartilhadas dos testes.

- `cfg`: config real (lê config/default.yaml do projeto) mas com os caminhos
  resolvidos para um diretório temporário, isolando os testes do disco real.
- `synthetic_atl06`: gera um .h5 no formato ATL06 com pontos válidos e
  inválidos conhecidos, para testar a extração sem baixar dado real.
"""

from pathlib import Path

import numpy as np
import h5py
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@pytest.fixture
def cfg(tmp_path):
    """Config base com paths sob tmp_path (não toca o disco real do projeto)."""
    from thwaites import load_config
    return load_config(base_dir=tmp_path, config_dir=CONFIG_DIR)


@pytest.fixture
def cfg_anual(tmp_path):
    from thwaites import load_config
    return load_config("anual", base_dir=tmp_path, config_dir=CONFIG_DIR)


@pytest.fixture
def synthetic_atl06(tmp_path, cfg):
    """
    Cria um .h5 estilo ATL06 com 2 feixes (gt1l, gt2r), cada um com 5 pontos:
      idx0: válido (h=500, qual=0)
      idx1: válido (h=510, qual=1)
      idx2: fill value (h=3.5e38)  -> descartado
      idx3: absurdo (h=-999)       -> descartado
      idx4: lat=NaN e qual=2       -> descartado
    O QC científico mantém apenas quality_summary=0: 1 por feixe, 2 no total.
    Retorna (caminho_h5, n_validos_esperado=2).
    """
    p = cfg.product
    epoch, spy = p.atlas_epoch_year, p.seconds_per_year
    delta = (2021.5 - epoch) * spy

    h5_path = tmp_path / "ATL06_20210715123456_12340312_007_01.h5"
    with h5py.File(h5_path, "w") as f:
        for beam_i, beam in enumerate(("gt1l", "gt2r")):
            g = f.create_group(f"{beam}/{p.segments_group}")
            g[p.variables["latitude"]]  = np.array([-75.0, -75.1, -75.2, -75.3, np.nan])
            g[p.variables["longitude"]] = np.array([-100.0, -100.1, -100.2, -100.3, -100.4])
            g[p.variables["height"]]    = np.array([500.0, 510.0, 3.5e38, -999.0, 520.0])
            g[p.variables["sigma"]]     = np.array([0.10, 0.12, 0.10, 0.10, 0.10])
            g[p.variables["delta_t"]]   = np.full(5, delta)
            g[p.variables["quality"]]   = np.array([0, 1, 0, 0, 2])
            g["segment_id"] = np.arange(100, 105, dtype=np.int32)
            track = f.create_group(f"{beam}/{p.segments_group}/ground_track")
            track["x_atc"] = np.arange(5, dtype=float) * 40.0
            track["y_atc"] = np.zeros(5)
            # posição real no ATL06 r007: raiz de land_ice_segments
            f[f"{beam}/{p.segments_group}"]["sigma_geo_h"] = np.full(5, 0.03)
            track["sigma_geo_at"] = np.full(5, 3.0)
            track["sigma_geo_xt"] = np.full(5, 3.0)
            fit = f.create_group(f"{beam}/{p.segments_group}/fit_statistics")
            fit["h_robust_sprd"] = np.full(5, 0.2)
            fit["n_fit_photons"] = np.full(5, 20, dtype=np.int32)
            fit["dh_fit_dx"] = np.full(5, 0.01)
            fit["h_expected_rms"] = np.full(5, 0.1)
            fit["snr_significance"] = np.zeros(5)
            # Grupo geophysical: tide_ocean com um fill em idx1 (valido).
            geo = f.create_group(f"{beam}/{p.segments_group}/{p.geophysical_group}")
            geo["tide_earth"] = np.zeros(5)
            geo["tide_load"] = np.zeros(5)
            geo["tide_ocean"]       = np.array(
                  [3.4e38 if beam_i == 0 else 0.5, 0.5, 0.5, 0.5, 0.5])
            geo["tide_pole"] = np.zeros(5)
            geo["tide_equilibrium"] = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
            geo["dac"]              = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
            geo["neutat_delay_total"] = np.zeros(5)
            geo["msw_flag"] = np.zeros(5, dtype=np.int8)
            geo["cloud_flg_asr"] = np.zeros(5, dtype=np.int8)
            geo["cloud_flg_atm"] = np.zeros(5, dtype=np.int8)
            geo["bsnow_conf"] = np.full(5, -1, dtype=np.int8)
            geo["solar_elevation"] = np.full(5, -10.0)
            # O geoide NÃO fica em `geophysical` no ATL06 real: fica em
            # `dem/geoid_h`. O fixture o criava como `geophysical/geoid`, que é
            # justamente o caminho errado que já produziu um campo 100% NaN em
            # todo o conjunto de dados. Criar aqui o caminho REAL faz o teste
            # exercer o mesmo acesso da produção — se o caminho da config
            # regredir, o teste cai.
            dem = f.create_group(f"{beam}/{p.segments_group}/dem")
            dem["geoid_h"]          = np.array([-30.0, -30.0, -30.0, -30.0, -30.0])
            dem["dem_h"]             = np.array([500.0, 500.0, 500.0, 500.0, 500.0])
    return h5_path, 2
