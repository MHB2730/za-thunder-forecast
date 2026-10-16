#!/usr/bin/env python3
"""Build TrailTether's South African wind-gust forecast from DWD ICON.

WHY THIS EXISTS
---------------
met.no publishes `wind_speed_of_gust` only inside its Nordic model. Probed live
on 2026-10-16: 0 of 90 hourly blocks at 13 South African points (Berg, Cape,
Cederberg, Swartberg, Otter, Amatola, Magaliesberg, Blyde, Durban), in both its
`complete` and `compact` products — and none at London, Berlin, Madrid or
Nairobi either, against 53 of 84 at Oslo. Every gust check in the app (the
62 km/h alert, the 62/75 km/h score vetoes, Trail Brain's wind kinds) therefore
either ran on MEAN wind in the gust slot or never fired at all. On 19 Oct 2026
the three global models agreed on 100-132 km/h gusts at Sani Pass while met.no's
mean there was 28 km/h: the app would have raised nothing.

This is the same hole `fetch_dwd_thunder.py` fills for thunder, filled from the
same model, licence and job.

WHAT IT PUBLISHES
-----------------
`za-gust.json.gz`: for every ICON cell in the South Africa box, the strongest
gust in each 3-hour block of each SAST day (km/h, rounded UP), on the SAME
cells and `grid_id` as the thunder feed, so the app reuses the thunder grid.

THE FIELD, CHECKED AGAINST THE LIVE FILES (2026-10-16)
------------------------------------------------------
`VMAX_10M` (shortName `max_i10fg`, m/s) is the maximum 10 m gust over the
**preceding ONE HOUR** at every step — `stepRange` 0-1, 1-2, 2-3, … 77-78, and
STILL one hour past +78 (80-81, 83-84, 89-90), where ICON global only outputs
3-hourly. Two consequences, both load-bearing:

  * Out to +78 h every hour is fetched. Sampling every third hour (as the
    thunder ladder does) would see one hour in three and miss two thirds of
    the gusts.
  * Past +78 h only one hour in three EXISTS. Those blocks are published with
    their honest coverage (1 of 3 hours), and the app must read them as
    "at least", never as the block's maximum.

AN AREA FIGURE, NOT A SUMMIT FIGURE
-----------------------------------
Each cell's value is the MEDIAN of all cells within 25 km — the reduction the
owner chose for the ensemble on 2026-08-31 (see spatial.py). Nearest-cell
sampling put Sani Pass on the windiest cell in the country (102.7 km/h against
30-38 from the other models); the median keeps real escarpment wind (93-124
km/h on 19 Oct) without that one-cell spike. The UI must say "around", and the
model's name.

FAIL-SAFE CONTRACT — same as the thunder feed
---------------------------------------------
  * A block with no decoded hour is `null` — UNKNOWN, never calm.
  * Coverage travels with the data (`cover` vs `expect`, hours per block), so a
    block that lost an hour to a failed download reads as "at least", not as
    the block's maximum.
  * Too few steps decoded overall: EXIT NON-ZERO and write nothing. The last
    good file stands and the app ages it out to unknown after 30 h.

Usage:
    python scripts/fetch_dwd_gust.py --out out/za-gust.json.gz
    python scripts/fetch_dwd_gust.py --run 2026101600 --steps 1,2,3
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
from fetch_dwd_thunder import (  # noqa: E402
    BASE, FETCH_ERRORS, LICENCE, SAST, SOURCE_URL, cell_index, fetch,
    grib_values, latest_run, log,
)
from spatial import DEFAULT_RADIUS_KM, KM_PER_DEG  # noqa: E402

# Hourly to +78 h (all that exists there), 3-hourly to +180 h after it.
DEFAULT_STEPS = list(range(1, 79)) + list(range(81, 181, 3))

UNKNOWN = 255  # a cell outside the mask; a block with no data is null instead
MAX_KMH = 254  # cap, so a real gust can never collide with UNKNOWN
BLOCK_H = 3
BLOCKS = 24 // BLOCK_H


def neighbours(lat: np.ndarray, lon: np.ndarray, radius_km: float,
               cache_dir: str) -> np.ndarray:
    """(n, k) indices of every cell within `radius_km` of each cell.

    Padded with `n`, which callers point at a NaN slot. Time-invariant, so it
    is cached next to the cell index.
    """
    n = lat.size
    cache = os.path.join(cache_dir, f"za_nb_{n}_{radius_km:g}km.npz")
    if os.path.exists(cache):
        return np.load(cache)["nb"]
    rows = []
    r2 = radius_km * radius_km
    for i in range(n):
        dy = (lat - lat[i]) * KM_PER_DEG
        dx = (lon - lon[i]) * KM_PER_DEG * np.cos(np.radians(lat[i]))
        rows.append(np.nonzero(dy * dy + dx * dx <= r2)[0])
    k = max(r.size for r in rows)
    nb = np.full((n, k), n, dtype=np.int32)
    for i, r in enumerate(rows):
        nb[i, : r.size] = r
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, nb=nb)
    sizes = [r.size for r in rows]
    log(f"  neighbours: {radius_km:g} km, {min(sizes)}-{k} cells "
        f"(median {int(np.median(sizes))})")
    return nb


def load_mask(path: str, n: int) -> np.ndarray:
    """The cells to publish — see build_gust_mask.py. Refuses a mask built for
    another grid, which would publish one place's gusts under another's cell."""
    m = json.load(open(path, encoding="utf-8"))
    if m["grid_id"] != f"icon-global-R03B07-za-{n}" or m["cell_count"] != n:
        raise SystemExit(f"FATAL: mask {path} is for {m['grid_id']}, grid has {n} "
                         "cells. Re-run build_gust_mask.py.")
    bits = np.frombuffer(base64.b64decode(m["keep_bits"]), dtype=np.uint8)
    return np.unpackbits(bits)[:n].astype(bool)


def area_median(values: np.ndarray, nb: np.ndarray) -> np.ndarray:
    ext = np.append(values.astype(np.float32), np.float32(np.nan))
    return np.nanmedian(ext[nb], axis=1)


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="ICON run, e.g. 2026101600 (default: latest)")
    ap.add_argument("--out", default="out/za-gust.json.gz")
    ap.add_argument("--cache", default=".dwd-cache")
    ap.add_argument("--steps", help="comma-separated forecast hours")
    ap.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM)
    ap.add_argument(
        "--mask",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "data", "za-gust-mask.json"),
    )
    ap.add_argument(
        "--raw",
        help="development: save the per-hour area medians here, or rebuild "
             "the file from them if it exists (no download)",
    )
    ap.add_argument(
        "--min-steps",
        type=int,
        default=90,
        help="refuse to publish below this many decoded steps (fail-safe)",
    )
    a = ap.parse_args()

    run = a.run or latest_run(dt.datetime.now(dt.timezone.utc))
    steps = [int(s) for s in a.steps.split(",")] if a.steps else DEFAULT_STEPS
    run_dt = dt.datetime.strptime(run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)

    log(f"DWD ICON global VMAX_10M — run {run} ({len(steps)} steps requested)")
    idx, lat, lon = cell_index(run, a.cache)
    nb = neighbours(lat, lon, a.radius_km, a.cache)
    n = idx.size
    grid_id = f"icon-global-R03B07-za-{n}"
    keep = load_mask(a.mask, n)  # before the downloads: a stale mask fails fast

    # The hours each block COULD carry: those inside this run's forecast
    # window [run, run + last step). An hour before the run started is not
    # missing data, it is the past.
    end = run_dt + dt.timedelta(hours=max(steps))
    expect: dict[str, list[int]] = {}
    t = run_dt
    while t < end:
        lt = t.astimezone(SAST)
        expect.setdefault(lt.strftime("%Y-%m-%d"), [0] * BLOCKS)[lt.hour // BLOCK_H] += 1
        t += dt.timedelta(hours=1)

    peak: dict[str, list[np.ndarray | None]] = {}
    cover: dict[str, list[int]] = {d: [0] * BLOCKS for d in expect}
    got = 0
    raw = dict(np.load(a.raw)) if a.raw and os.path.exists(a.raw) else None
    saved: dict[str, np.ndarray] = {}
    for s in steps:
        # The value at step s is the maximum over [s-1, s): file it under the
        # hour it covers, not the hour it is stamped with.
        start = (run_dt + dt.timedelta(hours=s - 1)).astimezone(SAST)
        key, b = start.strftime("%Y-%m-%d"), start.hour // BLOCK_H
        if raw is not None:
            if f"s{s}" not in raw:
                log(f"  +{s:03d}h not in {a.raw} — {key} block {b} short")
                continue
            kmh = raw[f"s{s}"]
        else:
            fn = f"icon_global_icosahedral_single-level_{run}_{s:03d}_VMAX_10M.grib2.bz2"
            try:
                vals = grib_values(fetch(f"{BASE}/{run[-2:]}/vmax_10m/{fn}"))
            except FETCH_ERRORS as e:
                log(f"  +{s:03d}h MISSING ({type(e).__name__}) — {key} block {b} short")
                continue
            kmh = np.ceil(area_median(vals[idx], nb) * 3.6)
            if a.raw:
                saved[f"s{s}"] = kmh.astype(np.float32)
        slot = peak.setdefault(key, [None] * BLOCKS)
        slot[b] = kmh if slot[b] is None else np.maximum(slot[b], kmh)
        cover[key][b] += 1
        got += 1
        if got % 20 == 0:
            log(f"  +{s:03d}h ok ({got}/{len(steps)})")

    if saved:
        np.savez_compressed(a.raw, **saved)
    log(f"decoded {got}/{len(steps)} steps")
    if got < a.min_steps:
        log(f"FATAL: only {got} steps decoded (min {a.min_steps}). "
            "Refusing to publish a partial file — the last good artifact stands.")
        return 1

    days = sorted(expect)

    def enc(v: np.ndarray) -> str:
        v = np.clip(np.nan_to_num(v, nan=UNKNOWN), 0, UNKNOWN)
        return b64(np.where(keep, np.minimum(v, MAX_KMH), UNKNOWN).astype(np.uint8))

    out_days = []
    for d in days:
        c, e = cover[d], expect[d]
        row = peak.get(d, [None] * BLOCKS)
        known = [v for b, v in enumerate(row) if v is not None and c[b] > 0]
        day = {"date": d, "cover": c, "expect": e}
        if not known:
            # Nothing decoded for this day: absent planes, so the app reads
            # UNKNOWN for it rather than a plane it might mistake for calm.
            day["full"] = False
        elif all(ci == ei for ci, ei in zip(c, e)):
            # Every hour the run covers was read: 3-hour blocks, each its
            # block's true maximum.
            day["full"] = True
            day["blocks"] = [
                enc(v) if v is not None and c[b] > 0 else None
                for b, v in enumerate(row)
            ]
        else:
            # Past +78 h (one hour in three exists) or a failed download: one
            # plane, the strongest hour read, which is a LOWER BOUND. Per-block
            # planes here would triple the file for figures that are only
            # "at least" anyway.
            day["full"] = False
            day["max"] = enc(np.max(np.stack(known), axis=0))
        out_days.append(day)

    doc = {
        "schema": 1,
        "model": "DWD ICON global (VMAX_10M — maximum 10 m wind gust)",
        "licence": LICENCE,
        "source": SOURCE_URL,
        "run": run_dt.isoformat(),
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timezone": "Africa/Johannesburg",
        "grid_id": grid_id,
        "cell_count": int(n),
        "unknown": UNKNOWN,
        "unit": "km/h",
        "rounding": "up, to the whole km/h",
        "scope": "area",
        "radius_km": a.radius_km,
        "reduction": f"median of all cells within {a.radius_km:g} km, per hour; "
                     "then the maximum over each block or day",
        "mask": "cells more than 30 km outside South Africa carry `unknown`",
        "block_hours": BLOCK_H,
        "steps_decoded": got,
        "steps_requested": len(steps),
        # Per day: hours read per block (`cover`) against hours the run could
        # give it (`expect`). `full` days carry 8 `blocks`; other days one
        # `max` plane that is a LOWER BOUND; a day with neither is unknown.
        "days": out_days,
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with gzip.open(a.out, "wb", compresslevel=9) as f:
        f.write(json.dumps(doc, separators=(",", ":")).encode("utf-8"))
    log(f"wrote {a.out}: {os.path.getsize(a.out)/1024:.1f} KiB gz "
        f"({int(keep.sum())} of {n} cells x {len(days)} days)")
    for day in out_days:
        planes = day.get("blocks") or [day.get("max")]
        vals = [np.frombuffer(base64.b64decode(x), dtype=np.uint8) for x in planes if x]
        top = max((int(v[v != UNKNOWN].max()) for v in vals if (v != UNKNOWN).any()),
                  default=None)
        log(f"  {day['date']}  {'full ' if day['full'] else 'lower'}  "
            f"cover {day['cover']} of {day['expect']}  max {top} km/h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
