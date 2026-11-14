"""get_closest_sounding picks the nearest station even when its launch hour
is not 00/12 UTC (Ny-Ålesund launches at ~11 UTC)."""

import sys
import types
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from pyradtran.io import RadiosondeAtmosphereGenerator

STATIONS = pd.DataFrame(
    {
        "id": ["SVM00001004", "SVM00001028"],
        "latitude": [78.9233, 74.5167],
        "longitude": [11.9222, 19.0047],
        "name": ["NY-ALESUND II", "BJORNOYA"],
        "last_year": [2100, 2100],
    }
)
LAUNCHES = {
    "SVM00001004": [datetime(2022, 3, 27, 11), datetime(2022, 3, 28, 11)],
    "SVM00001028": [datetime(2022, 3, 28, 0), datetime(2022, 3, 28, 12)],
}


class FakeIGRA:
    """Mimics siphon: exact match for a datetime, inclusive range for a pair."""

    @staticmethod
    def request_data(time, site_id):
        begin, end = (time, time) if isinstance(time, datetime) else time
        dates = [d for d in LAUNCHES[site_id] if begin <= d <= end]
        if not dates:
            raise ValueError("No dates match selection.")
        header = pd.DataFrame({"site_id": site_id, "date": dates})
        body = pd.DataFrame(
            {
                "date": [d for d in dates for _ in range(2)],
                "pressure": [1000.0, 500.0] * len(dates),
            }
        )
        return body, header


def test_nearest_station_with_off_synoptic_launch():
    fake = types.ModuleType("siphon.simplewebservice.igra2")
    fake.IGRAUpperAir = FakeIGRA
    modules = {
        "siphon": types.ModuleType("siphon"),
        "siphon.simplewebservice": types.ModuleType("siphon.simplewebservice"),
        "siphon.simplewebservice.igra2": fake,
    }
    with (
        patch.dict(sys.modules, modules),
        patch.object(
            RadiosondeAtmosphereGenerator, "get_station_list", return_value=STATIONS
        ),
    ):
        df, header, text = RadiosondeAtmosphereGenerator.get_closest_sounding(
            datetime(2022, 3, 28, 12), 78.92, 11.93
        )
    assert text.startswith("NY-ALESUND II at 2022-03-28 11:00 UTC")
    assert list(header["date"]) == [datetime(2022, 3, 28, 11)]
    assert len(df) == 2
