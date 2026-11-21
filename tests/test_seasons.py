from thwaites.glaciology.seasons import APPROVED_INTERVALS, decimal_year


def test_approved_season_durations_and_balance_set():
    duration = {item.name: item.duration_days for item in APPROVED_INTERVALS}
    assert duration["JJA-2020"] == 92
    assert duration["DJF-2020"] == 91
    assert duration["DJF-2021"] == 90
    primary = [item.name for item in APPROVED_INTERVALS
               if item.primary_comparison]
    assert primary == ["JJA-2020", "JJA-2021", "JJA-2022",
                       "DJF-2020", "DJF-2021", "DJF-2022"]


def test_decimal_year_respects_leap_year():
    item = next(x for x in APPROVED_INTERVALS if x.name == "JJA-2020")
    assert decimal_year(item.start) == 2020 + 152 / 366
