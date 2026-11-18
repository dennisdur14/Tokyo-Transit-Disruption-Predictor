"""Delay estimation from bus positions + timetable (no network needed)."""
from __future__ import annotations

import zipfile
from datetime import datetime

from src.collect import JST, estimate_delay, load_timetable

STOPS = [(1, 10 * 3600), (2, 10 * 3600 + 120), (3, 10 * 3600 + 300)]  # 10:00:00, 10:02:00, 10:05:00


def epoch(*args) -> int:
    return int(datetime(*args, tzinfo=JST).timestamp())


def test_delay_is_measured_from_the_midpoint_of_the_two_stops():
    # heading to stop 3 at 10:04:00; midpoint of 10:02 and 10:05 is 10:03:30 -> 30 s late
    assert estimate_delay(epoch(2026, 11, 18, 10, 4), 3, STOPS) == (30, 180)


def test_early_bus_has_negative_delay():
    assert estimate_delay(epoch(2026, 11, 18, 10, 0, 30), 2, STOPS)[0] == -30


def test_no_estimate_at_the_first_stop_or_for_unknown_stops():
    assert estimate_delay(epoch(2026, 11, 18, 10, 1), 1, STOPS) is None
    assert estimate_delay(epoch(2026, 11, 18, 10, 1), 9, STOPS) is None


def test_after_midnight_trip_belongs_to_the_previous_service_day():
    late = [(1, 24 * 3600 + 600), (2, 24 * 3600 + 1200)]  # "24:10" and "24:20" on the 18th's timetable
    assert estimate_delay(epoch(2026, 11, 19, 0, 16), 2, late)[0] == 60


def test_load_timetable_keeps_only_running_trips_and_normalises_names(tmp_path):
    path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("routes.txt", "﻿route_id,route_short_name\n006,都０１（Ｔ０１）\n")
        z.writestr("stop_times.txt", "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                                     "A,10:00:00,10:00:00,s1,1\nA,10:02:00,10:02:00,s2,2\n"
                                     "B,11:00:00,11:00:00,s1,1\n")
    routes, schedule = load_timetable(path, {"A"})
    assert routes == {"006": "都01(T01)"}
    assert schedule == {"A": [(1, 36000), (2, 36120)]}
