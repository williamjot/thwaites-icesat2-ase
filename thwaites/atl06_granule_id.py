"""Recuperação determinística da identidade orbital ATL06 pelo nome do arquivo.

Os Parquets ATL06 legacy deste projeto não guardam ``granule_id``, ``rgt``,
``cycle``, ``region`` nem ``delta_time`` como colunas. A única proveniência
sobrevivente está no nome do arquivo, herdado sem alteração do grânulo original::

    ATL06_YYYYMMDDHHMMSS_ttttccrr_RRR_VV.h5   ->   ...parquet

onde ``tttt`` é o RGT (Reference Ground Track), ``cc`` o número do ciclo de
repetição, ``rr`` a região orbital do grânulo, ``RRR`` o release e ``VV`` a
versão/revisão do processamento. É o padrão oficial NSIDC do ATL06.

Este módulo é deliberadamente **fail-closed**: só devolve identidade quando a
extração é inequívoca. Qualquer desvio do padrão — data inexistente no
calendário, RGT fora do intervalo físico da órbita, região fora de 1-14 —
levanta ``GranuleIdError`` em vez de devolver um valor plausível porém não
verificado.

Convenção temporal de ``t_year``
-------------------------------
``t_year`` nos Parquets legacy NÃO é ano decimal de calendário. A verificação
empírica (ver ``docs/LEGACY_GRANULE_ID_AUDIT.md``) mostra que ele é o tempo
ATLAS convertido em **anos julianos de 365,25 dias contados a partir da época
ATLAS SDP** (2018-01-01T00:00:00)::

    t_year = 2018 + delta_time / (365.25 * 86400)

Interpretar ``t_year`` como ano decimal de calendário introduz um erro
sistemático oscilatório de até ~0,75 dia (cerca de 18 h), com sinal variável ao
longo da missão. Por isso as conversões estão implementadas aqui, com o nome
explícito da convenção, e não devem ser refeitas ad hoc nos pipelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constantes da missão / do produto
# --------------------------------------------------------------------------- #

#: Época ATLAS SDP (``atlas_sdp_gps_epoch``), origem de ``delta_time`` no ATL06.
ATLAS_SDP_EPOCH = datetime(2018, 1, 1, tzinfo=timezone.utc)

#: Duração do ano juliano usada na definição de ``t_year`` legacy.
JULIAN_YEAR_DAYS = 365.25
JULIAN_YEAR_SECONDS = JULIAN_YEAR_DAYS * 86400.0

#: Número de RGTs da órbita de repetição de 91 dias do ICESat-2.
N_RGT = 1387

#: Regiões orbitais de granulação do ATL06 (1-14).
N_REGIONS = 14

#: Padrão oficial do nome do grânulo, sem extensão.
ATL06_STEM_PATTERN = re.compile(
    r"^ATL06_"
    r"(?P<timestamp>\d{14})_"
    r"(?P<rgt>\d{4})(?P<cycle>\d{2})(?P<region>\d{2})_"
    r"(?P<release>\d{3})_"
    r"(?P<version>\d{2})$"
)

_SUFFIX_PATTERN = re.compile(r"\.(parquet|h5|hdf5|nc)(\.\w+)?$", flags=re.IGNORECASE)


class GranuleIdError(ValueError):
    """O nome não permite recuperar a identidade orbital de forma inequívoca."""


@dataclass(frozen=True)
class GranuleId:
    """Identidade orbital recuperada do nome de um grânulo ATL06."""

    granule_filename: str
    granule_id: str
    acquisition_timestamp: datetime
    rgt: int
    cycle: int
    region: int
    release: int
    version: int

    @property
    def t_year(self) -> float:
        """Instante nominal do grânulo na mesma convenção do ``t_year`` legacy."""
        return datetime_to_t_year(self.acquisition_timestamp)

    @property
    def calendar_decimal_year(self) -> float:
        """Instante nominal como ano decimal de calendário (convenção distinta)."""
        return datetime_to_calendar_decimal_year(self.acquisition_timestamp)

    def as_record(self) -> dict:
        """Linha achatada para inventários CSV."""
        stamp = self.acquisition_timestamp
        return {
            "granule_filename": self.granule_filename,
            "granule_id": self.granule_id,
            "acquisition_timestamp": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "year": stamp.year,
            "month": stamp.month,
            "day": stamp.day,
            "doy": int(stamp.strftime("%j")),
            "rgt": self.rgt,
            "cycle": self.cycle,
            "region": self.region,
            "release": self.release,
            "version": self.version,
            "nominal_t_year": self.t_year,
        }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _parse_timestamp(digits: str) -> datetime:
    """Converte ``YYYYMMDDHHMMSS`` em datetime UTC, rejeitando datas inexistentes."""
    try:
        stamp = datetime.strptime(digits, "%Y%m%d%H%M%S")
    except ValueError as exc:  # 20190231..., 20190601256199, ...
        raise GranuleIdError(
            f"timestamp inválido no nome do grânulo: {digits}") from exc
    if not 2018 <= stamp.year <= 2100:
        raise GranuleIdError(
            f"timestamp fora da vida útil do ICESat-2 (lançado em 2018): {digits}")
    return stamp.replace(tzinfo=timezone.utc)


def parse_granule_stem(stem: str) -> GranuleId:
    """Recupera a identidade orbital do nome-base (sem extensão) de um grânulo.

    Levanta ``GranuleIdError`` sempre que a extração não for inequívoca.
    """
    match = ATL06_STEM_PATTERN.fullmatch(stem)
    if match is None:
        raise GranuleIdError(
            f"nome fora do padrão oficial ATL06 (identidade orbital ausente): {stem}")

    timestamp = _parse_timestamp(match.group("timestamp"))
    rgt = int(match.group("rgt"))
    cycle = int(match.group("cycle"))
    region = int(match.group("region"))
    release = int(match.group("release"))
    version = int(match.group("version"))

    if not 1 <= rgt <= N_RGT:
        raise GranuleIdError(
            f"RGT {rgt:04d} fora do intervalo 1-{N_RGT} da órbita de 91 dias: {stem}")
    if cycle < 1:
        raise GranuleIdError(f"ciclo de repetição inválido ({cycle}): {stem}")
    if not 1 <= region <= N_REGIONS:
        raise GranuleIdError(
            f"região orbital {region} fora do intervalo 1-{N_REGIONS}: {stem}")
    if release < 1:
        raise GranuleIdError(f"release inválido ({release}): {stem}")

    return GranuleId(
        granule_filename=stem,
        granule_id=stem,
        acquisition_timestamp=timestamp,
        rgt=rgt,
        cycle=cycle,
        region=region,
        release=release,
        version=version,
    )


def parse_granule_path(path: str | Path) -> GranuleId:
    """Como ``parse_granule_stem``, aceitando caminho ou nome com extensão."""
    name = Path(path).name
    stem = _SUFFIX_PATTERN.sub("", name)
    return parse_granule_stem(stem)


def try_parse_granule_path(path: str | Path) -> GranuleId | None:
    """Versão não-levantadora: devolve ``None`` quando a extração é ambígua."""
    try:
        return parse_granule_path(path)
    except GranuleIdError:
        return None


# --------------------------------------------------------------------------- #
# Conversões temporais
# --------------------------------------------------------------------------- #

def datetime_to_t_year(stamp: datetime) -> float:
    """Instante UTC -> ``t_year`` legacy (anos julianos desde a época ATLAS)."""
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    seconds = (stamp - ATLAS_SDP_EPOCH).total_seconds()
    return 2018.0 + seconds / JULIAN_YEAR_SECONDS


def t_year_to_datetime(t_year: float) -> datetime:
    """``t_year`` legacy -> instante UTC, na convenção de ano juliano ATLAS."""
    seconds = (float(t_year) - 2018.0) * JULIAN_YEAR_SECONDS
    return ATLAS_SDP_EPOCH + timedelta(seconds=seconds)


def t_year_to_delta_time(t_year: float) -> float:
    """``t_year`` legacy -> segundos desde a época ATLAS SDP (proxy de delta_time).

    É um *proxy*: reconstrói o valor numérico de ``delta_time``, mas não recupera
    a convenção de escala de tempo (GPS/TAI vs UTC) do arquivo HDF5 original, que
    não está mais disponível localmente.
    """
    return (float(t_year) - 2018.0) * JULIAN_YEAR_SECONDS


def datetime_to_calendar_decimal_year(stamp: datetime) -> float:
    """Instante UTC -> ano decimal de calendário (ano civil como unidade)."""
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    year_start = datetime(stamp.year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(stamp.year + 1, 1, 1, tzinfo=timezone.utc)
    elapsed = (stamp - year_start).total_seconds()
    length = (year_end - year_start).total_seconds()
    return stamp.year + elapsed / length


def t_year_to_calendar_decimal_year(t_year):
    """Vetorizado: ``t_year`` legacy -> ano decimal de **calendário**.

    Necessário sempre que o eixo temporal de um produto externo for ano decimal
    civil — o IMAU-FDM, por exemplo. Confundir as duas convenções desloca a
    consulta em até ~0,75 dia, que é menos que o passo do FDM (10 dias), mas o
    erro é sistemático e gratuito de evitar.
    """
    import numpy as np

    t = np.asarray(t_year, dtype="float64")
    segundos = (t - 2018.0) * JULIAN_YEAR_SECONDS
    epoch = np.datetime64("2018-01-01T00:00:00", "s")
    instante = epoch + segundos.astype("timedelta64[s]")
    ano = instante.astype("datetime64[Y]")
    inicio = ano.astype("datetime64[s]")
    fim = (ano + np.timedelta64(1, "Y")).astype("datetime64[s]")
    decorrido = (instante - inicio).astype("float64")
    duracao = (fim - inicio).astype("float64")
    return ano.astype(int) + 1970 + decorrido / duracao


def calendar_misinterpretation_error_seconds(t_year: float) -> float:
    """Erro, em segundos, de ler ``t_year`` como se fosse ano decimal de calendário.

    Positivo quando a leitura ingênua adianta o instante verdadeiro.
    """
    true_time = t_year_to_datetime(t_year)
    year = int(t_year)
    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    naive_time = year_start + timedelta(
        seconds=(t_year - year) * (year_end - year_start).total_seconds())
    return (naive_time - true_time).total_seconds()


__all__ = [
    "ATL06_STEM_PATTERN",
    "ATLAS_SDP_EPOCH",
    "GranuleId",
    "GranuleIdError",
    "JULIAN_YEAR_DAYS",
    "JULIAN_YEAR_SECONDS",
    "N_REGIONS",
    "N_RGT",
    "calendar_misinterpretation_error_seconds",
    "datetime_to_calendar_decimal_year",
    "datetime_to_t_year",
    "parse_granule_path",
    "parse_granule_stem",
    "t_year_to_datetime",
    "t_year_to_delta_time",
    "try_parse_granule_path",
]
