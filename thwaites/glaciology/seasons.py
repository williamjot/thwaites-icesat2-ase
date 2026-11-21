"""Intervalos sazonais aprovados e ciclos ATL06 de suporte."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class SeasonalInterval:
    name: str
    season: str
    comparison_year: int
    start: datetime
    end: datetime
    start_cycles: tuple[int, int]
    end_cycles: tuple[int, int]
    primary_comparison: bool

    @property
    def duration_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400.0


def _utc(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc)


APPROVED_INTERVALS = (
    SeasonalInterval("JJA-2019", "JJA", 2019, _utc(2019, 6, 1),
                     _utc(2019, 9, 1), (3, 4), (4, 5), False),
    SeasonalInterval("JJA-2020", "JJA", 2020, _utc(2020, 6, 1),
                     _utc(2020, 9, 1), (7, 8), (8, 9), True),
    SeasonalInterval("JJA-2021", "JJA", 2021, _utc(2021, 6, 1),
                     _utc(2021, 9, 1), (11, 12), (12, 13), True),
    SeasonalInterval("JJA-2022", "JJA", 2022, _utc(2022, 6, 1),
                     _utc(2022, 9, 1), (15, 16), (16, 17), True),
    SeasonalInterval("DJF-2020", "DJF", 2020, _utc(2019, 12, 1),
                     _utc(2020, 3, 1), (5, 6), (6, 7), True),
    SeasonalInterval("DJF-2021", "DJF", 2021, _utc(2020, 12, 1),
                     _utc(2021, 3, 1), (9, 10), (10, 11), True),
    SeasonalInterval("DJF-2022", "DJF", 2022, _utc(2021, 12, 1),
                     _utc(2022, 3, 1), (13, 14), (14, 15), True),
)


def decimal_year(value: datetime) -> float:
    start = _utc(value.year, 1, 1)
    end = _utc(value.year + 1, 1, 1)
    return value.year + (value - start).total_seconds() / (
        end - start).total_seconds()
