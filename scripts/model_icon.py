#!/usr/bin/env python3
"""DWD ICON global — the four daily veto fields, per point.

CC BY 4.0. ⚠️ Attribution is a LICENCE CONDITION and must reach the UI.

WHY ICON CARRIES THE ENSEMBLE
-----------------------------
It is the only model here that publishes a WMO weather code (`WW`, the same
0-99 space the app is built on), so it is the only one that can vote on the two
vetoes GFS and ECMWF abstain from: thunder and snow. `fetch_dwd_thunder.py`
exists because of exactly that gap — met.no publishes thunder only in its
Nordic model, so in South Africa the veto for what the app calls the #1
mountain killer is otherwise dead.

That makes ICON's vote the least redundant of the three, and worth its cost.

⚠️ AND IT IS THE EXPENSIVE ONE. ICON global has no subregion or variable
filter: every field is the whole globe at ~3.3 MB compressed. Four variables on
a 3-hourly ladder to +120 h is ~530 MB per run, against ~2 KB per step for GFS
and ~2.5 MB for ECMWF. That is the price of the only thunder vote, but it is
also why the step ladder should not be lengthened casually.

WHY THERE IS NO met.no ADAPTER IN THIS DIRECTORY
------------------------------------------------
met.no is the primary source and its verdict is already computed by
`metno.js evaluateDay` in the browser — by the authority itself. Re-deriving it
server-side would mean porting `metNoSymbolToWmo` as a second place to drift,
and it could not vote on gusts anyway: checked live 2026-08-31 at Cathedral
Peak, the `complete` product returns `wind_speed` and NO `wind_speed_of_gust`,
so a server-side port would have to substitute a different quantity for the
gust veto and would disagree with the client for reasons that are not the
model's.

So the page combines met.no's own client-side verdict with the three
server-side models in this feed. Fewer moving parts, one less port to drift,
and met.no's vote is computed by met.no's own implementation.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_dwd_thunder import (  # noqa: E402
    BASE, FETCH_ERRORS, SAST, cell_index, fetch, grib_values, latest_run,
)
from spatial import (  # noqa: E402
    DEFAULT_RADIUS_KM, cells_within_km, describe, reduce_max, reduce_median,
)

MODEL = "DWD ICON global"
LICENCE = "Deutscher Wetterdienst (DWD), ICON — CC BY 4.0"
SOURCE_URL = "https://opendata.dwd.de"

# (directory, FILENAME token). Verified present on the live server 2026-08-31.
VARS = {
    "ww": ("ww", "WW"),
    "t2m": ("t_2m", "T_2M"),
    "precip": ("tot_prec", "TOT_PREC"),
    "gust": ("vmax_10m", "VMAX_10M"),
}


def _url(run: str, step: int, var: str) -> str:
    d, token = VARS[var]
    return (f"{BASE}/{run[-2:]}/{d}/"
            f"icon_global_icosahedral_single-level_{run}_{step:03d}_{token}.grib2.bz2")


def fetch_daily(
    points: list[tuple[float, float]],
    *,
    run: str | None = None,
    hours: int = 120,
    cache: str = ".dwd-cache",
    radius_km: float = DEFAULT_RADIUS_KM,
    log=print,
) -> dict:
    """Daily precip sum, max gust, min temp and worst WMO code, bucketed to SAST.

    Same fail-safe contract as the other adapters and as the thunder feed: a day
    is `complete` only if every step covering it decoded, in every variable. A
    partial day must render as unknown, never as a quiet all-clear.
    """
    run = run or latest_run(dt.datetime.now(dt.timezone.utc))
    run_dt = dt.datetime.strptime(run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    idx, lat, lon = cell_index(run, cache)
    # ONE reduction, shared with every other model — see spatial.py. Replaced
    # nearest-neighbour, which put Sani Pass on the windiest cell in the country.
    sel = cells_within_km(points, lat, lon, radius_km)

    steps = list(range(3, hours + 1, 3))
    per_day: dict[str, dict] = {}
    expected: dict[str, int] = {}
    prev_precip: list[float] | None = None
    got = 0

    for s in steps:
        valid = (run_dt + dt.timedelta(hours=s)).astimezone(SAST)
        key = valid.strftime("%Y-%m-%d")
        expected[key] = expected.get(key, 0) + 1

        vals: dict[str, list[float]] = {}
        missing = None
        for var in VARS:
            try:
                arr = grib_values(fetch(_url(run, s, var)))
            except FETCH_ERRORS as e:
                missing = f"{var} ({type(e).__name__})"
                break
            vals[var] = arr[idx]
        if missing:
            log(f"  ICON +{s:03d}h MISSING {missing} — {key} -> incomplete")
            # A gap breaks the precipitation difference across it.
            prev_precip = None
            continue

        d = per_day.setdefault(key, {
            "precip_sum": [0.0] * len(points),
            "wind_gust_max": [0.0] * len(points),
            "temp_min": [None] * len(points),
            "weather_code": [0] * len(points),
            "steps": 0,
        })
        for i in range(len(points)):
            # ⚠️ TOT_PREC accumulates from the start of the run, in kg m-2 (= mm).
            # Consecutive steps must be differenced; summing the reported value
            # multiplies the day's total by the step count. Verified against the
            # live file rather than assumed — the same check that caught GFS's
            # overlapping windows and ECMWF's metres.
            cur = reduce_median(vals["precip"], sel[i])
            inc = cur if prev_precip is None else max(0.0, cur - prev_precip[i])
            d["precip_sum"][i] += inc
            # VMAX_10M is a maximum over the preceding interval, in m/s.
            d["wind_gust_max"][i] = max(
                d["wind_gust_max"][i], reduce_median(vals["gust"], sel[i]) * 3.6)
            c = reduce_median(vals["t2m"], sel[i]) - 273.15
            d["temp_min"][i] = c if d["temp_min"][i] is None else min(d["temp_min"][i], c)
            # MAX, not median: thunder 15 km away is a real hazard on a ridge.
            w = int(reduce_max(vals["ww"], sel[i]))
            if w > d["weather_code"][i]:
                d["weather_code"][i] = w
        prev_precip = [reduce_median(vals["precip"], sel[i]) for i in range(len(points))]
        d["steps"] += 1
        got += 1
        if got % 5 == 0:
            log(f"  ICON +{s:03d}h ok ({got}/{len(steps)})")

    for key, d in per_day.items():
        d["complete"] = d["steps"] == expected.get(key, 0)

    return {
        "model": MODEL,
        "licence": LICENCE,
        "source": SOURCE_URL,
        "run": run_dt.isoformat(),
        "steps_decoded": got,
        "steps_requested": len(steps),
        # The reason ICON is here: it is the only model that can vote on these.
        "thunder_assessed": True,
        "snow_assessed": True,
        "spatial": describe(sel, radius_km),
        "days": per_day,
    }
