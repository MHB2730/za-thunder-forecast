#!/usr/bin/env python3
"""ECMWF IFS 0.25° open data — the four daily veto fields, per point.

CC BY 4.0. ⚠️ Attribution is a LICENCE CONDITION, not a courtesy: it must reach
the UI, exactly as for DWD.

WHY BYTE RANGES AND NOT A PLAIN DOWNLOAD
----------------------------------------
ECMWF publishes no subregion or variable filter. Measured live 2026-08-31, one
forecast step of the global 0.25° operational run is **146 MB** — about 5.8 GB
per run across a 3-hourly ladder to +120 h, which is not a reasonable thing to
do to a free service four times a day.

But every step has a sibling `.index` file (~40 KB) of JSON lines carrying
`_offset` and `_length` per GRIB message:

    {"param": "tp", "levtype": "sfc", "step": "12", "_offset": 0, "_length": 813573}

So the three fields we need are fetched with HTTP Range requests — roughly
2.5 MB per step instead of 146 MB, a ~60x saving. `Range` support was verified
against the live server before this was written.

⚠️ ECMWF RATE-LIMITS. A 429 was observed on the second request of a probe on
2026-08-31 and cleared on retry, so the backoff below is load-bearing rather
than defensive boilerplate. Be conservative about adding steps.

⚠️ THE ENSEMBLE MEASURES MODEL DISAGREEMENT, SO THE REDUCTION MUST BE IDENTICAL
------------------------------------------------------------------------------
ECMWF publishes fields GFS does not — `mn2t3` (true minimum 2 m temperature
over the preceding 3 h), `mucape`, `ptype`, `sf`. Using them here would make
ECMWF's daily minimum systematically colder than GFS's, because a true minimum
catches the trough between samples and an instantaneous reading does not.

That difference would show up on the page as ECMWF disagreeing with GFS about
frost — and it would be an artefact of THIS FILE, not of the models. The whole
product claim is "how many independent models condemn this day"; a reduction
that differs per model quietly turns that into "how consistently did Matt's
pipeline sample each model", which is worthless and, on a safety panel,
misleading.

So: instantaneous `2t` sampled on the SAME 3-hourly ladder as every other
model. `mn2t3` is deliberately unused. If the reduction is ever improved, it
must be improved for every model in the same commit.

`mucape` is likewise unused. CAPE is not a WMO weather code, and turning it
into one would mean inventing a convective risk index and defending it — the
exact thing `fetch_dwd_thunder.py` chose ICON's `WW` field to avoid. ECMWF
therefore abstains on thunder, like GFS.
"""

from __future__ import annotations

import datetime as dt
import http.client
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

import eccodes as ec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial import (  # noqa: E402
    DEFAULT_RADIUS_KM, cells_within_km, describe, reduce_max, reduce_median,
)

BASE = "https://data.ecmwf.int/forecasts"
UA = "TrailTether/2.0 (https://trailtether.app contact@hilltrek.co.za)"
MODEL = "ECMWF IFS 0.25° (open data)"
LICENCE = "ECMWF IFS open data — CC BY 4.0"
SOURCE_URL = "https://data.ecmwf.int"

SAST = dt.timezone(dt.timedelta(hours=2))

# `10fg` is the maximum gust over the preceding period; `2t` and `tp` are the
# instantaneous and accumulated fields. Names verified against a live .index.
WANT = ("tp", "10fg", "2t")

FETCH_ERRORS = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    http.client.HTTPException,
    OSError,
)


def _get(url: str, rng: tuple[int, int] | None = None,
         timeout: int = 90, attempts: int = 4) -> bytes:
    """GET with optional byte range and backoff.

    The backoff exists for ECMWF's 429s specifically — see the module note.
    """
    last = None
    for i in range(attempts):
        try:
            headers = {"User-Agent": UA}
            if rng is not None:
                headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 and i < attempts - 1:
                time.sleep(2 ** i)
                continue
            if i == attempts - 1:
                raise
        except FETCH_ERRORS as e:
            last = e
    raise RuntimeError(f"giving up on {url}: {last}")


def _stem(run_day: str, cycle: int, step: int) -> str:
    return (f"{BASE}/{run_day}/{cycle:02d}z/ifs/0p25/oper/"
            f"{run_day}{cycle:02d}0000-{step}h-oper-fc")


def _sample_messages(raw: bytes, points: list[tuple[float, float]]) -> dict:
    out: dict[str, list[float]] = {}
    fd, path = tempfile.mkstemp(suffix=".grib2")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        with open(path, "rb") as f:
            while True:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    break
                try:
                    name = ec.codes_get(gid, "shortName")
                    vals = (
                        ec.codes_get_array(gid, 'values'),
                        ec.codes_get_array(gid, 'latitudes'),
                        ec.codes_get_array(gid, 'longitudes'),
                    )
                    out.setdefault(name, vals)
                finally:
                    ec.codes_release(gid)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return out


def fetch_step(run_day: str, cycle: int, step: int,
               points: list[tuple[float, float]]) -> dict[str, list[float]]:
    """One step: read the index, range-fetch only the fields we need."""
    stem = _stem(run_day, cycle, step)
    idx = _get(stem + ".index").decode("utf-8", "replace")
    spans: list[tuple[str, int, int]] = []
    for line in idx.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("levtype") == "sfc" and d.get("param") in WANT:
            spans.append((d["param"], int(d["_offset"]), int(d["_length"])))

    got: dict[str, list[float]] = {}
    for param, off, length in spans:
        if param in got:
            continue
        raw = _get(stem + ".grib2", rng=(off, off + length - 1))
        got.update({k: v for k, v in _sample_messages(raw, points).items()
                    if k not in got})
    return got


def fetch_daily(
    points: list[tuple[float, float]],
    *,
    run_day: str | None = None,
    cycle: int = 0,
    hours: int = 120,
    radius_km: float = DEFAULT_RADIUS_KM,
    log=print,
) -> dict:
    """Daily precip sum, max gust and min temp per point, bucketed to SAST.

    Same fail-safe contract as the other adapters: a day is `complete` only if
    every step covering it decoded.
    """
    now = dt.datetime.now(dt.timezone.utc)
    if run_day is None:
        run_day = (now - dt.timedelta(days=1)).strftime("%Y%m%d")
    run_dt = dt.datetime.strptime(run_day, "%Y%m%d").replace(
        hour=cycle, tzinfo=dt.timezone.utc
    )

    steps = list(range(3, hours + 1, 3))
    sel = None
    prev_tp: list[float] | None = None
    per_day: dict[str, dict] = {}
    expected: dict[str, int] = {}
    got = 0

    for s in steps:
        valid = (run_dt + dt.timedelta(hours=s)).astimezone(SAST)
        key = valid.strftime("%Y-%m-%d")
        expected[key] = expected.get(key, 0) + 1
        try:
            f = fetch_step(run_day, cycle, s, points)
        except Exception as e:  # noqa: BLE001
            log(f"  ECMWF +{s:03d}h MISSING ({type(e).__name__}) — {key} -> incomplete")
            prev_tp = None  # the next difference would span a gap
            continue

        tp, gust, t2 = f.get("tp"), f.get("10fg"), f.get("2t")
        if tp is not None and sel is None:
            sel = cells_within_km(points, tp[1], tp[2], radius_km)
        if tp is None or gust is None or t2 is None:
            log(f"  ECMWF +{s:03d}h short — {key} -> incomplete")
            prev_tp = None
            continue

        d = per_day.setdefault(key, {
            "precip_sum": [0.0] * len(points),
            "wind_gust_max": [0.0] * len(points),
            "temp_min": [None] * len(points),
            "steps": 0,
        })
        tp_pt = [reduce_median(tp[0], sel[i]) for i in range(len(points))]
        gust_pt = [reduce_median(gust[0], sel[i]) for i in range(len(points))]
        t2_pt = [reduce_median(t2[0], sel[i]) for i in range(len(points))]
        for i in range(len(points)):
            # ⚠️ ECMWF `tp` accumulates from the START OF THE RUN and is in
            # METRES, not mm. Both are easy to get wrong in the direction that
            # invents rain: forgetting the difference multiplies a day's total
            # by the number of steps, and forgetting the unit multiplies by
            # 1000. The empirical check is in the commit message.
            cur_m = tp_pt[i]
            inc_m = cur_m if prev_tp is None else max(0.0, cur_m - prev_tp[i])
            d["precip_sum"][i] += inc_m * 1000.0
            d["wind_gust_max"][i] = max(d["wind_gust_max"][i], gust_pt[i] * 3.6)
            c = t2_pt[i] - 273.15
            d["temp_min"][i] = c if d["temp_min"][i] is None else min(d["temp_min"][i], c)
        prev_tp = tp_pt
        d["steps"] += 1
        got += 1
        if got % 5 == 0:
            log(f"  ECMWF +{s:03d}h ok ({got}/{len(steps)})")

    for key, d in per_day.items():
        d["complete"] = d["steps"] == expected.get(key, 0)

    return {
        "model": MODEL,
        "licence": LICENCE,
        "source": SOURCE_URL,
        "run": run_dt.isoformat(),
        "steps_decoded": got,
        "steps_requested": len(steps),
        # No WMO code used — see the module note on mucape and ptype.
        "thunder_assessed": False,
        "snow_assessed": False,
        "spatial": describe(sel, radius_km) if sel else None,
        "days": per_day,
    }
