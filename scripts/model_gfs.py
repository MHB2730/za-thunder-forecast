#!/usr/bin/env python3
"""NOAA GFS 0.25° — the four daily veto fields, per point.

Public domain (NOAA/NCEP), so no attribution condition, though the ensemble
names it anyway because a reader deserves to know which models voted.

WHY GFS IS THE CHEAP ONE
------------------------
NOMADS' `filter_gfs_0p25_1hr.pl` supports SUBREGION filtering, so a South
Africa box comes back as roughly 2 KB per step with three variables in it.
Contrast ICON global, which has no such filter: every field is the whole globe
at ~3.3 MB compressed. Adding GFS costs almost nothing.

⚠️ WHAT GFS DOES NOT GIVE US: a WMO weather code.
GFS publishes categorical precipitation flags (CRAIN/CSNOW/CFRZR/CICEP), not
the 0-99 WMO code the veto ladder reads. Deriving one would mean inventing a
translation and defending it — precisely what fetch_dwd_thunder.py's header
says was avoided by choosing ICON's WW field.

So this adapter reports `weather_code = None` and `thunder_assessed = False`,
and the ladder's "Lightning not assessed" rung handles the consequence
honestly. GFS votes on wind, rain and cold; it abstains on thunder and snow.
The ensemble must present that as an abstention, NOT as an all-clear — see the
`thunder_assessed` note in veto.py.

⚠️ THE ACCUMULATION TRAP — READ BEFORE CHANGING `hours` OR THE STEP LADDER
--------------------------------------------------------------------------
GFS resets its `tp` accumulator every 6 hours, and each file carries BOTH a
since-reset window and a since-run window. Observed live 2026-08-31:

    f003 -> ['0-3',   '0-3']       f012 -> ['6-12',  '0-12']
    f006 -> ['0-6',   '0-6']       f015 -> ['12-15', '0-15']
    f009 -> ['6-9',   '0-9']       f024 -> ['18-24', '0-24']

So `0-3` and `0-6` OVERLAP. Summing the short window at every 3-hourly step
double-counts the first three hours of every six-hour block — silently, and in
the direction that INVENTS rainfall. Rain drives both the `p > 12` veto and the
wet-cold hypothermia trap, so that would manufacture hazards.

The fix is the subtraction in `_incremental()`: within a block the first step's
window is already incremental, and the second must have the first subtracted.
`test_gfs_accumulation.py` pins it against the real window strings.
"""

from __future__ import annotations

import datetime as dt
import http.client
import os
import sys
import tempfile
import urllib.error
import urllib.request

import eccodes as ec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial import (  # noqa: E402
    DEFAULT_RADIUS_KM, cells_within_km, describe, reduce_max, reduce_median,
)

FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
UA = "TrailTether/2.0 (https://trailtether.app contact@hilltrek.co.za)"
MODEL = "NOAA GFS 0.25°"
LICENCE = "NOAA/NCEP GFS — public domain (17 U.S.C. §105)"
SOURCE_URL = "https://nomads.ncep.noaa.gov"

# SAST, matching fetch_dwd_thunder.py. A UTC day boundary would put a 01:00
# SAST storm on the wrong date.
SAST = dt.timezone(dt.timedelta(hours=2))

FETCH_ERRORS = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    http.client.HTTPException,
    OSError,
)


def block_start(step: int) -> int:
    """First hour of the 6-hourly accumulation block containing `step`.

    GFS resets at 0/6/12/18, and a step landing exactly ON a boundary belongs
    to the block that ENDS there — f006 carries '0-6', not '6-6'. Hence the
    `- 1`, which is the whole reason this is a named function with a test
    rather than an inline expression.
    """
    if step <= 0:
        return 0
    return ((step - 1) // 6) * 6


def _incremental(step: int, values: dict[int, float]) -> float | None:
    """Rain that fell in the 3 hours ENDING at `step`.

    `values[s]` is the since-block-start accumulation reported at step `s`.
    """
    b = block_start(step)
    cur = values.get(step)
    if cur is None:
        return None
    prev_step = step - 3
    if prev_step <= b:
        # First step of the block: its window already starts at the reset.
        return cur
    prev = values.get(prev_step)
    if prev is None:
        return None
    # Clamp: accumulations are monotonic within a block, but a decode glitch
    # must not produce negative rainfall, which would MASK a wet day.
    return max(0.0, cur - prev)


def _url(run_day: str, cycle: int, step: int, bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    return (
        f"{FILTER}?file=gfs.t{cycle:02d}z.pgrb2.0p25.f{step:03d}"
        "&var_APCP=on&var_GUST=on&var_TMP=on"
        "&lev_surface=on&lev_2_m_above_ground=on"
        f"&subregion=&leftlon={w}&rightlon={e}&toplat={n}&bottomlat={s}"
        f"&dir=%2Fgfs.{run_day}%2F{cycle:02d}%2Fatmos"
    )


def _fetch(url: str, timeout: int = 90, attempts: int = 3) -> bytes:
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            if not raw.startswith(b"GRIB"):
                # NOMADS answers 200 with an HTML error page when a run is not
                # yet published. Treating that as data would decode to nothing
                # and read as a step with no weather in it.
                raise ValueError(f"not GRIB ({len(raw)} bytes)")
            return raw
        except (ValueError, *FETCH_ERRORS) as e:  # noqa: B030
            last = e
    raise RuntimeError(f"giving up on {url}: {last}")


def _sample(raw: bytes, points: list[tuple[float, float]]) -> dict[str, list[float]]:
    """Nearest-neighbour sample of every message, keyed by shortName.

    Returns {shortName: [value per point]}. For `tp` the key carries the window
    so the caller can pick the incremental one — see the accumulation note.
    """
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
                    lvl = ec.codes_get(gid, "typeOfLevel")
                    rng = ec.codes_get(gid, "stepRange")
                    # `t` at surface is SKIN temperature; the ladder wants the
                    # 2 m air temperature, which arrives as `2t` on
                    # heightAboveGround. Taking `t` would read several degrees
                    # warm on a clear night — in the direction that hides frost.
                    if name == "t" and lvl != "heightAboveGround":
                        continue
                    key = f"{name}|{rng}" if name == "tp" else name
                    vals = (
                        ec.codes_get_array(gid, 'values'),
                        ec.codes_get_array(gid, 'latitudes'),
                        ec.codes_get_array(gid, 'longitudes'),
                    )
                    # Keep the SHORTEST window when a name repeats: NOMADS
                    # sends tp twice, and the since-run window is the wrong one.
                    if key not in out:
                        out[key] = vals
                finally:
                    ec.codes_release(gid)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return out


def _pick_tp(sampled: dict[str, list[float]], step: int) -> list[float] | None:
    """The since-block-start accumulation for this step, not the since-run one."""
    want = f"tp|{block_start(step)}-{step}"
    if want in sampled:
        return sampled[want]
    # f003/f006 report '0-3'/'0-6', which the expression above already builds.
    return None


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

    Fail-safe, matching fetch_dwd_thunder.py's contract: a day is only marked
    `complete` if every step covering it decoded. An incomplete day must be
    rendered as unknown, never as a quiet all-clear.
    """
    now = dt.datetime.now(dt.timezone.utc)
    if run_day is None:
        # Yesterday's cycle is certainly published; today's may still be
        # uploading. The ensemble is a day-scale product, so preferring a
        # complete older run to a half-published newer one is the right trade.
        run_day = (now - dt.timedelta(days=1)).strftime("%Y%m%d")
    run_dt = dt.datetime.strptime(run_day, "%Y%m%d").replace(
        hour=cycle, tzinfo=dt.timezone.utc
    )

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    pad = 1.0
    bbox = (min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad)

    steps = list(range(3, hours + 1, 3))
    # Computed once from the first decoded grid: the grid is constant across
    # steps, and recomputing per step would be wasted work, not safety.
    sel = None
    tp_acc: dict[int, list[float]] = {}
    per_day: dict[str, dict] = {}
    expected: dict[str, int] = {}
    got = 0

    for s in steps:
        valid = (run_dt + dt.timedelta(hours=s)).astimezone(SAST)
        key = valid.strftime("%Y-%m-%d")
        expected[key] = expected.get(key, 0) + 1
        try:
            sampled = _sample(_fetch(_url(run_day, cycle, s, bbox)), points)
        except Exception as e:  # noqa: BLE001 - one bad step demotes one day
            log(f"  GFS +{s:03d}h MISSING ({type(e).__name__}) — {key} -> incomplete")
            continue

        tp = _pick_tp(sampled, s)
        gust = sampled.get("gust")
        t2 = sampled.get("2t")
        if tp is not None and sel is None:
            # SAME kilometre-radius reduction as every other model — spatial.py.
            sel = cells_within_km(points, tp[1], tp[2], radius_km)
        if tp is None or gust is None or t2 is None:
            log(f"  GFS +{s:03d}h short (tp={tp is not None} "
                f"gust={gust is not None} 2t={t2 is not None}) — {key} -> incomplete")
            continue
        # Reduce the grids to one representative value per point BEFORE any
        # arithmetic, so the accumulation subtraction operates on the same
        # quantity it will be reported as.
        tp_pt = [reduce_median(tp[0], sel[i]) for i in range(len(points))]
        gust_pt = [reduce_median(gust[0], sel[i]) for i in range(len(points))]
        t2_pt = [reduce_median(t2[0], sel[i]) for i in range(len(points))]
        tp_acc[s] = tp_pt

        d = per_day.setdefault(key, {
            "precip_sum": [0.0] * len(points),
            "wind_gust_max": [0.0] * len(points),
            "temp_min": [None] * len(points),
            "steps": 0,
        })
        for i in range(len(points)):
            # Goes through the SAME _incremental() the accumulation test pins.
            # Inlining the subtraction here instead would leave the test
            # exercising a function production does not call, which is how a
            # green test and a double-counted rainfall coexist.
            add = _incremental(s, {k: v[i] for k, v in tp_acc.items()})
            if add is not None:
                d["precip_sum"][i] += add
            # m/s -> km/h
            d["wind_gust_max"][i] = max(d["wind_gust_max"][i], gust_pt[i] * 3.6)
            # K -> °C
            c = t2_pt[i] - 273.15
            d["temp_min"][i] = c if d["temp_min"][i] is None else min(d["temp_min"][i], c)
        d["steps"] += 1
        got += 1
        if got % 10 == 0:
            log(f"  GFS +{s:03d}h ok ({got}/{len(steps)})")

    for key, d in per_day.items():
        d["complete"] = d["steps"] == expected.get(key, 0)

    return {
        "model": MODEL,
        "licence": LICENCE,
        "source": SOURCE_URL,
        "run": run_dt.isoformat(),
        "steps_decoded": got,
        "steps_requested": len(steps),
        # GFS has no WMO code, so it abstains on thunder and snow. Stated in the
        # payload rather than assumed by the consumer.
        "thunder_assessed": False,
        "snow_assessed": False,
        "spatial": describe(sel, radius_km) if sel else None,
        "days": per_day,
    }
