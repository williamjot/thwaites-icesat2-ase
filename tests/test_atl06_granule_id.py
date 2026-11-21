"""Testes do parser de identidade orbital ATL06 (nomes reais do repositório)."""

from datetime import datetime, timezone

import pytest

from thwaites.atl06_granule_id import (
    ATLAS_SDP_EPOCH,
    GranuleIdError,
    calendar_misinterpretation_error_seconds,
    datetime_to_calendar_decimal_year,
    datetime_to_t_year,
    parse_granule_path,
    parse_granule_stem,
    t_year_to_datetime,
    t_year_to_delta_time,
    try_parse_granule_path,
)

# Nomes copiados verbatim de data/jja/processed e data/djf/processed.
REAL_NAMES = [
    "ATL06_20190601094826_09790312_007_01.parquet",
    "ATL06_20190601111700_09800311_007_01.parquet",
    "ATL06_20190101050015_00570210_007_01.parquet",
    "ATL06_20250831233902_11792812_007_01.parquet",
    "ATL06_20190602231759_10030310_007_01.parquet",
]


@pytest.mark.parametrize("name", REAL_NAMES)
def test_nomes_reais_sao_parseaveis(name):
    gid = parse_granule_path(name)
    assert gid.release == 7
    assert gid.version == 1
    assert 1 <= gid.rgt <= 1387
    assert gid.cycle >= 1
    assert 1 <= gid.region <= 14
    assert gid.acquisition_timestamp.tzinfo is timezone.utc


def test_campos_extraidos_do_nome_jja():
    gid = parse_granule_path("ATL06_20190601094826_09790312_007_01.parquet")
    assert gid.granule_id == "ATL06_20190601094826_09790312_007_01"
    assert gid.acquisition_timestamp == datetime(2019, 6, 1, 9, 48, 26,
                                                 tzinfo=timezone.utc)
    assert (gid.rgt, gid.cycle, gid.region) == (979, 3, 12)
    assert (gid.release, gid.version) == (7, 1)


def test_campos_extraidos_do_nome_djf():
    gid = parse_granule_path("ATL06_20190101050015_00570210_007_01.parquet")
    assert (gid.rgt, gid.cycle, gid.region) == (57, 2, 10)
    assert gid.acquisition_timestamp == datetime(2019, 1, 1, 5, 0, 15,
                                                 tzinfo=timezone.utc)


def test_caminho_completo_e_extensao_h5_equivalem():
    base = "ATL06_20250831233902_11792812_007_01"
    a = parse_granule_path(f"D:/dados/data/jja/processed/{base}.parquet")
    b = parse_granule_path(f"/tmp/{base}.h5")
    c = parse_granule_stem(base)
    assert a == b == c


def test_as_record_tem_as_colunas_do_inventario():
    rec = parse_granule_path(REAL_NAMES[0]).as_record()
    esperadas = {"granule_filename", "granule_id", "acquisition_timestamp", "year",
                 "month", "day", "doy", "rgt", "cycle", "region", "release",
                 "version", "nominal_t_year"}
    assert esperadas <= set(rec)
    assert (rec["year"], rec["month"], rec["day"], rec["doy"]) == (2019, 6, 1, 152)


# --------------------------------------------------------------------------- #
# Fail-closed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stem", [
    "ATL06_20190601094826_09790312_007",          # falta a versão
    "ATL06_2019060109482_09790312_007_01",        # timestamp com 13 dígitos
    "ATL06_20190601094826_0979031_007_01",        # bloco orbital com 7 dígitos
    "ATL03_20190601094826_09790312_007_01",       # produto errado
    "atl06_20190601094826_09790312_007_01",       # caixa diferente
    "ATL06_20190601094826_09790312_007_01_extra",
    "consolidado_jja.parquet",
])
def test_nomes_fora_do_padrao_sao_rejeitados(stem):
    with pytest.raises(GranuleIdError):
        parse_granule_stem(stem)


@pytest.mark.parametrize("stem", [
    "ATL06_20190231094826_09790312_007_01",   # 31 de fevereiro
    "ATL06_20190601254826_09790312_007_01",   # hora 25
    "ATL06_20170601094826_09790312_007_01",   # antes do lançamento
    "ATL06_20190601094826_13880312_007_01",   # RGT 1388 > 1387
    "ATL06_20190601094826_09790300_007_01",   # região 0
    "ATL06_20190601094826_09790315_007_01",   # região 15
    "ATL06_20190601094826_09790012_007_01",   # ciclo 0
    "ATL06_20190601094826_09790312_000_01",   # release 0
])
def test_valores_impossiveis_sao_rejeitados(stem):
    with pytest.raises(GranuleIdError):
        parse_granule_stem(stem)


def test_try_parse_devolve_none_em_vez_de_levantar():
    assert try_parse_granule_path("nao_e_um_granulo.parquet") is None
    assert try_parse_granule_path(REAL_NAMES[0]) is not None


# --------------------------------------------------------------------------- #
# Convenção temporal de t_year
# --------------------------------------------------------------------------- #

def test_t_year_e_ano_juliano_desde_a_epoca_atlas():
    # Um ano juliano exato após a época.
    assert datetime_to_t_year(ATLAS_SDP_EPOCH) == pytest.approx(2018.0)
    um_ano = t_year_to_datetime(2019.0)
    assert (um_ano - ATLAS_SDP_EPOCH).total_seconds() == pytest.approx(365.25 * 86400)


def test_roundtrip_datetime_t_year():
    stamp = datetime(2021, 7, 14, 3, 25, 51, tzinfo=timezone.utc)
    volta = t_year_to_datetime(datetime_to_t_year(stamp))
    assert abs((volta - stamp).total_seconds()) < 1e-3


def test_t_year_reproduz_o_valor_observado_no_parquet_legacy():
    """Valor medido em data/jja/processed/ATL06_20190601094826_09790312_007_01.

    O menor t_year do arquivo é 2019.413849781056; o nome declara
    2019-06-01T09:48:26Z. A diferença tem de ficar dentro de poucos segundos.
    """
    gid = parse_granule_path("ATL06_20190601094826_09790312_007_01.parquet")
    observado = 2019.413849781056
    erro_s = (observado - gid.t_year) * 365.25 * 86400
    assert abs(erro_s) < 5.0


def test_delta_time_proxy_bate_com_a_epoca():
    stamp = datetime(2019, 6, 1, 9, 48, 26, tzinfo=timezone.utc)
    esperado = (stamp - ATLAS_SDP_EPOCH).total_seconds()
    assert t_year_to_delta_time(datetime_to_t_year(stamp)) == pytest.approx(
        esperado, abs=1e-3)


def test_leitura_como_ano_de_calendario_e_uma_convencao_diferente():
    stamp = datetime(2019, 6, 1, 9, 48, 26, tzinfo=timezone.utc)
    juliano = datetime_to_t_year(stamp)
    calendario = datetime_to_calendar_decimal_year(stamp)
    assert juliano != pytest.approx(calendario, abs=1e-6)
    # O erro de confundir as duas convenções é de horas, não de segundos.
    erro_s = calendar_misinterpretation_error_seconds(juliano)
    assert 3600 < abs(erro_s) < 0.76 * 86400
