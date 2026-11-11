#!/usr/bin/env python3
"""Build TrailTether's South African 10 m wind forecast from DWD ICON.

WHY THIS EXISTS
---------------
TrailTether's map draws moving wind streaks over the whole country. That needs
a wind VECTOR field (direction as well as speed) on a grid, hour by hour, which
met.no's point forecasts cannot supply cheaply. This is the same model, run,
licence, cells and job as `fetch_dwd_thunder.py` and `fetch_dwd_gust.py`.

THE FIELDS, CHECKED AGAINST THE LIVE FILES (2026-11-11, run 2026111106 +0 h)
---------------------------------------------------------------------------
  * `U_10M` — directory `u_10m/`, shortName `10u`, "10 metre U wind
    component", m s**-1, heightAboveGround 10, packingType grid_ccsds.
  * `V_10M` — directory `v_10m/`, shortName `10v`, "10 metre V wind
    component", same units, level and packing.
  * Both carry 2,949,120 values: the ICON global R03B07 grid the thunder cell
    index is built on. u is towards the east, v towards the north (on the
    icosahedral grid there is no grid-relative rotation to undo).
  * Instantaneous values at the step time (stepRange `0m` at +0 h), hourly
    to +78 h — so every hour asked for here exists.

WHAT IT PUBLISHES
-----------------
`za-wind.json.gz`: every hour from +0 to +30 h of the run. DWD publishes a run
3-4 h after its nominal time and the app may hold a file for hours after that,
so +30 h keeps "now" through roughly +24 h on the phone.

Every cell of the thunder grid (same order, same `grid_id`, so the app reuses
`za-thunder-grid.json.gz` for the coordinates). No area median — the wind field
is smooth and the streaks want the model's own field — and no gust mask.

JSON SCHEMA (schema 1; gzipped, compact separators)
---------------------------------------------------
  {
    "schema": 1,
    "model": "DWD ICON global (U_10M, V_10M — 10 m wind)",
    "licence": "Deutscher Wetterdienst (DWD), ICON — CC BY 4.0",
    "source": "https://opendata.dwd.de",
    "run": "2026-11-11T06:00:00+00:00",        # ISO, UTC
    "generated": "2026-11-11T12:01:02.345+00:00",
    "grid_id": "icon-global-R03B07-za-13024",  # == za-thunder-grid.json.gz
    "cell_count": 13024,
    "unit": "m/s",
    "scale": 0.5,             # m/s per int8 step: value = int8 * scale
    "clamp": 63.5,            # |component| above this is stored as ±63.5
    "steps_requested": 31,
    "steps_decoded": 31,
    "steps": [0, 1, 2, ...],  # forecast hours PRESENT, ascending; a step
                              # that failed to decode is absent, never zeroed
    "valid": ["2026-11-11T06:00:00+00:00", ...],  # UTC, one per step
    "u": ["<b64>", ...],      # one per step: cell_count int8, eastward
    "v": ["<b64>", ...]       # one per step: cell_count int8, northward
  }

Each u/v string is base64 of `cell_count` signed bytes (int8, two's
complement) in thunder-grid cell order. Decode: `int8 * scale` m/s.

FAIL-SAFE CONTRACT — same as the thunder and gust feeds
-------------------------------------------------------
  * A step whose u OR v did not download, decode, match the grid or come back
    finite is left OUT of `steps` — the app sees a gap, never a fake calm hour.
  * Too few steps decoded (default: under 80 % of those requested): EXIT
    NON-ZERO and write nothing. The last good file stands in R2.
  * A field whose size is not the ICON R03B07 global grid, or a published
    thunder grid whose grid_id / cell_count differ from the cell index, stops
    the build: wind for one cell must never be drawn at another's position.

Usage:
    python scripts/fetch_dwd_wind.py --out out/za-wind.json.gz
    python scripts/fetch_dwd_wind.py --run 2026111106 --steps 0,1,2
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_dwd_thunder as thunder  # noqa: E402
from fetch_dwd_thunder import (  # noqa: E402
    BASE, FETCH_ERRORS, LICENCE, SOURCE_URL, latest_run, log,
)

DEFAULT_STEPS = list(range(0, 31))  # +0 .. +30 h, every hour

# Same floor as the gust feed: a few lost hours cost smoothness, not honesty;
# a mostly-failed run must not replace a good file.
MIN_SHARE = 0.8

SCALE = 0.5   # m/s per int8 step
QMAX = 127    # ±127 * 0.5 = ±63.5 m/s; -128 unused so the range is symmetric
CLAMP = QMAX * SCALE

# Values in one ICON global R03B07 field, read from the live U_10M/V_10M files
# (numberOfValues) on 2026-11-11. The cell index points into this array; a
# field of any other size means DWD changed the grid and `idx` would pick the
# wrong cells.
ICON_GLOBAL_VALUES = 2_949_120

# DWD directory and file token per component. Verified against the live
# directory listings of u_10m/ and v_10m/ on 2026-11-11.
FIELDS = {"u": ("u_10m", "U_10M"), "v": ("v_10m", "V_10M")}


class GridMismatch(Exception):
    """The data does not belong to the grid the cell index describes."""


def url(run: str, step: int, comp: str) -> str:
    d, token = FIELDS[comp]
    return (f"{BASE}/{run[-2:]}/{d}/"
            f"icon_global_icosahedral_single-level_{run}_{step:03d}_{token}.grib2.bz2")


def quantise(ms: np.ndarray) -> np.ndarray:
    """m/s -> int8 at SCALE, clamped to ±CLAMP. Caller ensures finite input."""
    return np.clip(np.rint(np.asarray(ms, dtype=np.float64) / SCALE),
                   -QMAX, QMAX).astype(np.int8)


def dequantise(q: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * SCALE


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def check_grid_file(path: str, n: int) -> None:
    """Refuse to build against a published thunder grid for other cells."""
    with gzip.open(path, "rb") as f:
        g = json.loads(f.read().decode("utf-8"))
    want = f"icon-global-R03B07-za-{n}"
    if g.get("grid_id") != want or g.get("cell_count") != n:
        raise GridMismatch(
            f"{path} is grid {g.get('grid_id')} ({g.get('cell_count')} cells); "
            f"the cell index is {want}. Rebuild the thunder grid or clear the cache.")


def field(run: str, step: int, comp: str, idx: np.ndarray) -> np.ndarray:
    vals = thunder.grib_values(thunder.fetch(url(run, step, comp)))
    if vals.size != ICON_GLOBAL_VALUES:
        raise GridMismatch(
            f"{comp} +{step:03d}h has {vals.size} values, expected "
            f"{ICON_GLOBAL_VALUES} (ICON R03B07)")
    return vals[idx]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="ICON run, e.g. 2026111106 (default: latest)")
    ap.add_argument("--out", default="out/za-wind.json.gz")
    ap.add_argument("--cache", default=".dwd-cache")
    ap.add_argument("--steps", help="comma-separated forecast hours (default 0..30)")
    ap.add_argument(
        "--grid",
        help="published thunder grid to check grid_id against (default: "
             "za-thunder-grid.json.gz next to --out, if present)",
    )
    ap.add_argument(
        "--min-steps",
        type=int,
        help="refuse to publish below this many decoded steps "
             f"(default {MIN_SHARE:.0%} of those requested)",
    )
    a = ap.parse_args(argv)

    run = a.run or latest_run(dt.datetime.now(dt.timezone.utc))
    steps = [int(s) for s in a.steps.split(",")] if a.steps else list(DEFAULT_STEPS)
    steps = sorted(set(steps))
    min_steps = (a.min_steps if a.min_steps is not None
                 else int(np.ceil(MIN_SHARE * len(steps))))
    run_dt = dt.datetime.strptime(run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)

    log(f"DWD ICON global U_10M/V_10M — run {run} ({len(steps)} steps requested)")
    idx, _lat, _lon = thunder.cell_index(run, a.cache)
    n = int(idx.size)
    grid_id = f"icon-global-R03B07-za-{n}"

    # Before the downloads, so a stale cache fails fast.
    grid_path = a.grid or os.path.join(
        os.path.dirname(os.path.abspath(a.out)), "za-thunder-grid.json.gz")
    try:
        if os.path.exists(grid_path):
            check_grid_file(grid_path, n)
            log(f"  grid: {grid_path} matches {grid_id}")
        elif a.grid:
            log(f"FATAL: --grid {a.grid} not found")
            return 2
        else:
            log(f"  grid: no {grid_path} to check against (cell index {grid_id})")
    except GridMismatch as e:
        log(f"FATAL: {e}")
        return 2

    got_steps: list[int] = []
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    for s in steps:
        try:
            u = field(run, s, "u", idx)
            v = field(run, s, "v", idx)
        except GridMismatch as e:
            log(f"FATAL: {e}. Refusing to publish — the last good artifact stands.")
            return 2
        except FETCH_ERRORS as e:
            log(f"  +{s:03d}h MISSING ({type(e).__name__}) — step left out")
            continue
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            log(f"  +{s:03d}h has non-finite values — step left out")
            continue
        got_steps.append(s)
        us.append(quantise(u))
        vs.append(quantise(v))
        if len(got_steps) % 10 == 0:
            log(f"  +{s:03d}h ok ({len(got_steps)}/{len(steps)})")

    got = len(got_steps)
    log(f"decoded {got}/{len(steps)} steps")
    if got < min_steps:
        log(f"FATAL: only {got} steps decoded (min {min_steps}). "
            "Refusing to publish a partial file — the last good artifact stands.")
        return 1

    doc = {
        "schema": 1,
        "model": "DWD ICON global (U_10M, V_10M — 10 m wind)",
        "licence": LICENCE,
        "source": SOURCE_URL,
        "run": run_dt.isoformat(),
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "grid_id": grid_id,
        "cell_count": n,
        "unit": "m/s",
        "scale": SCALE,
        "clamp": CLAMP,
        "steps_requested": len(steps),
        "steps_decoded": got,
        "steps": got_steps,
        "valid": [(run_dt + dt.timedelta(hours=s)).isoformat() for s in got_steps],
        "u": [b64(q) for q in us],
        "v": [b64(q) for q in vs],
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with gzip.open(a.out, "wb", compresslevel=9) as f:
        f.write(json.dumps(doc, separators=(",", ":")).encode("utf-8"))
    log(f"wrote {a.out}: {os.path.getsize(a.out)/1024:.1f} KiB gz "
        f"({n} cells x {got} steps)")
    for s, qu, qv in zip(got_steps, us, vs):
        sp = np.hypot(dequantise(qu), dequantise(qv))
        log(f"  +{s:02d}h  speed min {sp.min():.1f}  mean {sp.mean():.1f}  "
            f"max {sp.max():.1f} m/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
