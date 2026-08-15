#!/usr/bin/env python3
"""Build TrailTether's South African thunderstorm forecast from DWD ICON.

WHY THIS EXISTS
---------------
The hiking score has a thunderstorm veto that caps a day at 2.0 — the app's own
comment calls lightning "the #1 mountain killer". In South Africa that veto was
DEAD, and silently so: it keys on `weatherCode >= 95 || thunderProb > 15`, and
met.no supplies neither here.

Probed live 2026-08-06 across 16 points: met.no publishes
`probability_of_thunder` only inside its Nordic model (53-54 hourly blocks at
each of eight Nordic controls, ZERO at each of Cathedral Peak, Sani Top,
Durban, Cape Town, Congo, Manaus, Jakarta and Kampala), and it derives its
`*andthunder` SYMBOLS from that same field — so outside the Nordics no thunder
symbol is ever emitted either, Miami in peak August storm season included.
Both halves of the test are permanently false in the target region.

DWD's ICON global model fills exactly that hole:
  * field `WW` = "weather interpretation (WMO)" — the SAME WMO code space the
    app is already built on (0-99, thunder at 95/96/99). No translation layer,
    no risk index to invent and defend.
  * genuinely global. Verified 2026-08-06 at +12h: 72,651 thunder cells
    worldwide, distributed where convection actually was — India 3,791
    (monsoon), Florida/Caribbean 776, Congo 382, Nordics 193, Amazon 141. That
    is the opposite of met.no's home-region-only behaviour.
  * ~13 km spacing. Nearest cell to Cathedral Peak sits 0.78 km away.
  * free, CC BY 4.0 (commercial use permitted WITH attribution — see LICENCE
    below; the attribution is not optional and must reach the UI).
  * four runs a day (00/06/12/18 UTC), out to +180 h.

WHY A PYTHON SCRIPT AND NOT A SUPABASE EDGE FUNCTION
----------------------------------------------------
The GRIB2 files are `packingType = grid_ccsds` (data representation template
42) — CCSDS/AEC adaptive entropy coding. There is no decoder for that in the
Deno ecosystem worth betting a safety feature on, so decoding has to happen
somewhere with eccodes. This script is designed to run on a schedule in CI and
publish a small static artifact the app can cache offline.

FAIL-SAFE CONTRACT — READ BEFORE EDITING
-----------------------------------------
This whole feature exists because an absent reading was being presented as a
confident all-clear. This script must not repeat that mistake at the ingest
layer. Therefore:

  * A day is only marked ASSESSED if every forecast step covering it was
    downloaded and decoded. Any gap and the day is emitted as UNKNOWN.
  * `code = 0` means "the model forecast no thunder". `code = UNKNOWN` (255)
    means "we could not tell". They are different values on purpose, and the
    app must render them differently.
  * If too few steps survive overall, the script EXITS NON-ZERO and writes
    nothing rather than publishing a partial file that reads as a quiet
    region-wide all-clear. A stale-but-honest artifact beats a fresh lie.

Usage:
    python scripts/fetch_dwd_thunder.py --out out/za-thunder.json.gz
    python scripts/fetch_dwd_thunder.py --run 2026080600 --steps 0,3,6
"""

import argparse
import base64
import bz2
import datetime as dt
import gzip
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

try:
    import eccodes as ec
except ImportError:  # pragma: no cover - environment problem, not logic
    sys.exit("eccodes is required: pip install eccodes (plus the ecCodes library)")

BASE = "https://opendata.dwd.de/weather/nwp/icon/grib"
UA = "TrailTether/2.0 (https://trailtether.app contact@hilltrek.co.za)"

# CC BY 4.0 requires attribution wherever the data is shown. This string is
# carried in the artifact so the app cannot ship the data without it.
LICENCE = "Deutscher Wetterdienst (DWD), ICON — CC BY 4.0"
SOURCE_URL = "https://opendata.dwd.de"

# South Africa, matching kZaBoundsSW/kZaBoundsNE in
# trailtether_app/lib/core/constants.dart. Kept in step with those on purpose:
# a cell the app can select but the grid does not cover would read as unknown
# forever.
ZA_SW = (-34.85213, 16.45065)
ZA_NE = (-22.12381, 32.91835)

# ICON global is hourly to +78 h and 3-hourly thereafter, so a 3-hourly ladder
# exists across the whole range. 3 h is finer than the day-level reduction
# needs, but it is what makes "first thunder hour" meaningful — the Berg's
# signature is a clear morning and a storm by 15:00, and a coarser sample
# would blur exactly that.
DEFAULT_STEPS = list(range(0, 181, 3))

# SAST. The app buckets days with .toLocal() and CI pins TZ=Africa/Johannesburg
# for the same reason — a UTC day boundary would put a 01:00 SAST storm on the
# wrong date and shift an afternoon storm's hour by two.
SAST = dt.timezone(dt.timedelta(hours=2))

UNKNOWN = 255  # sentinel in both the code and hour planes — never a real value
NO_HOUR = 254  # assessed, but no thunder that day, so no first-thunder hour

THUNDER_CODES = (95, 96, 99)


def log(msg: str) -> None:
    print(msg, flush=True)


# Everything a failed download can raise. `http.client.IncompleteRead` is an
# HTTPException, NOT an OSError — caught the hard way when a real truncated
# read (48 KiB of an expected 671 KiB) crashed the whole run instead of
# demoting one day to UNKNOWN. Missing that distinction would turn a transient
# network blip into a total outage of the thunder feed.
FETCH_ERRORS = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    http.client.HTTPException,
    OSError,
)


def fetch(url: str, timeout: int = 120, attempts: int = 3) -> bytes:
    """GET with retries. Length-checked: a short read must raise, not return.

    A truncated GRIB2 body would either fail to decode or — worse — decode to
    something wrong. Neither may be allowed to reach the artifact silently.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                declared = r.headers.get("Content-Length")
                body = r.read()
            if declared is not None and len(body) != int(declared):
                raise http.client.IncompleteRead(body, int(declared) - len(body))
            return body
        except FETCH_ERRORS as e:
            last = e
            if i + 1 < attempts:
                time.sleep(2 * (i + 1))
    raise last  # type: ignore[misc]


def grib_values(raw_bz2: bytes) -> np.ndarray:
    """Decode one bz2-wrapped GRIB2 message to its value array."""
    h = ec.codes_new_from_message(bz2.decompress(raw_bz2))
    try:
        return ec.codes_get_values(h)
    finally:
        ec.codes_release(h)


def latest_run(now: dt.datetime) -> str:
    """Most recent ICON run that is plausibly published.

    DWD posts a run roughly 3-4 h after its nominal time. Being conservative
    here costs a few hours of freshness; being optimistic costs a 404 storm and
    a half-built artifact, which the fail-safe contract above then throws away.
    """
    t = now - dt.timedelta(hours=4)
    return f"{t:%Y%m%d}{(t.hour // 6) * 6:02d}"


def cell_index(run: str, cache_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (indices_into_full_grid, lat, lon) for cells inside ZA.

    clat/clon are time-invariant, so this is computed once and cached. After the
    first run every cycle only needs the WW files.
    """
    cache = os.path.join(cache_dir, "za_cells.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"  cell index: cached ({z['idx'].size} cells)")
        return z["idx"], z["lat"], z["lon"]

    coords = {}
    for name in ("clat", "clon"):
        listing = fetch(f"{BASE}/{run[-2:]}/{name}/").decode("utf-8", "replace")
        import re

        files = re.findall(r'href="(icon_global[^"]+\.grib2\.bz2)"', listing)
        if not files:
            raise RuntimeError(f"no {name} file listed for run {run}")
        log(f"  {name}: {files[0]}")
        coords[name] = grib_values(fetch(f"{BASE}/{run[-2:]}/{name}/{files[0]}"))

    lat, lon = coords["clat"], coords["clon"]
    inside = (
        (lat >= ZA_SW[0]) & (lat <= ZA_NE[0]) & (lon >= ZA_SW[1]) & (lon <= ZA_NE[1])
    )
    idx = np.nonzero(inside)[0].astype(np.int32)
    log(f"  cell index: {idx.size} cells inside ZA of {lat.size} global")

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, idx=idx, lat=lat[idx], lon=lon[idx])
    return idx, lat[idx], lon[idx]


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="ICON run, e.g. 2026080600 (default: latest)")
    ap.add_argument("--out", default="out/za-thunder.json.gz")
    ap.add_argument("--cache", default=".dwd-cache")
    ap.add_argument("--steps", help="comma-separated forecast hours (default 0..180/3)")
    ap.add_argument(
        "--min-steps",
        type=int,
        default=40,
        help="refuse to publish below this many decoded steps (fail-safe)",
    )
    # Exists so the thunder path can be exercised for real. Over South Africa in
    # winter every cell is legitimately 0, so a run here proves only that the
    # script can emit zeros — which is indistinguishable from a broken decode,
    # and "silently emits a clean all-clear" is the exact failure this whole
    # feature was built to remove. Point it at a convective region (India in
    # August, say) and the same code path must come back non-zero.
    ap.add_argument(
        "--bbox",
        help="south,west,north,east — defaults to South Africa. For testing.",
    )
    a = ap.parse_args()

    global ZA_SW, ZA_NE
    if a.bbox:
        s, w, n_, e = (float(x) for x in a.bbox.split(","))
        ZA_SW, ZA_NE = (s, w), (n_, e)
        log(f"bbox override: {ZA_SW} .. {ZA_NE}")

    run = a.run or latest_run(dt.datetime.now(dt.timezone.utc))
    steps = (
        [int(s) for s in a.steps.split(",")] if a.steps else list(DEFAULT_STEPS)
    )
    run_dt = dt.datetime.strptime(run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)

    log(f"DWD ICON global WW — run {run} ({len(steps)} steps requested)")
    idx, lat, lon = cell_index(run, a.cache)

    # day (SAST) -> {step_hour_sast: code array over ZA cells}
    per_day: dict[str, dict[int, np.ndarray]] = {}
    expected: dict[str, int] = {}
    got = 0

    for s in steps:
        valid = (run_dt + dt.timedelta(hours=s)).astimezone(SAST)
        key = valid.strftime("%Y-%m-%d")
        expected[key] = expected.get(key, 0) + 1

        fn = f"icon_global_icosahedral_single-level_{run}_{s:03d}_WW.grib2.bz2"
        url = f"{BASE}/{run[-2:]}/ww/{fn}"
        try:
            vals = grib_values(fetch(url))
        except FETCH_ERRORS as e:
            log(f"  +{s:03d}h MISSING ({type(e).__name__}) — day {key} -> UNKNOWN")
            continue

        per_day.setdefault(key, {})[valid.hour] = vals[idx].astype(np.uint8)
        got += 1
        if got % 10 == 0:
            log(f"  +{s:03d}h ok ({got}/{len(steps)})")

    log(f"decoded {got}/{len(steps)} steps")
    if got < a.min_steps:
        # Fail-safe: publishing here would put a region-wide "no thunder" on the
        # wire, which is precisely the fabricated all-clear this feature exists
        # to remove. Keep whatever is already published instead.
        log(
            f"FATAL: only {got} steps decoded (min {a.min_steps}). "
            "Refusing to publish a partial file — the last good artifact stands."
        )
        return 1

    n = idx.size
    days, codes, hours, assessed = [], [], [], []
    for key in sorted(per_day):
        by_hour = per_day[key]
        # Only trust a day we sampled completely. A day missing steps could be
        # missing exactly the convective afternoon.
        complete = len(by_hour) == expected[key]

        code = np.zeros(n, dtype=np.uint8)
        hour = np.full(n, NO_HOUR, dtype=np.uint8)
        if complete:
            for h in sorted(by_hour):
                v = by_hour[h]
                is_t = np.isin(v, THUNDER_CODES)
                # Worst code of the day, and the FIRST hour thunder appears —
                # "off the summit by 13:00" is the actionable half.
                code = np.where(is_t & (v > code), v, code)
                hour = np.where(is_t & (hour == NO_HOUR), h, hour)
        else:
            code[:] = UNKNOWN
            hour[:] = UNKNOWN
            log(f"  {key}: {len(by_hour)}/{expected[key]} steps -> UNKNOWN")

        days.append(key)
        codes.append(b64(code))
        hours.append(b64(hour))
        assessed.append(bool(complete))

    doc = {
        "schema": 1,
        "model": "DWD ICON global (WW — weather interpretation, WMO)",
        "licence": LICENCE,
        "source": SOURCE_URL,
        "run": run_dt.isoformat(),
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timezone": "Africa/Johannesburg",
        "steps_decoded": got,
        "steps_requested": len(steps),
        # Sentinels are part of the contract with the app, so they travel with
        # the data rather than living only in a Dart constant that can drift.
        "unknown": UNKNOWN,
        "no_hour": NO_HOUR,
        "thunder_codes": list(THUNDER_CODES),
        "cell_count": int(n),
        "days": days,
        "assessed": assessed,
        "code": codes,
        "hour": hours,
    }

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    blob = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    with gzip.open(a.out, "wb", compresslevel=9) as f:
        f.write(blob)

    # The cell coordinates are TIME-INVARIANT and dominate the payload — about
    # 35 KiB of the artifact, versus a couple of KiB for the forecast planes
    # (which are mostly one repeated value and so compress to almost nothing).
    # Re-sending them four times a day would be the bulk of the download for
    # data that has not changed. They go in their own file the app fetches once
    # and keeps; `grid_id` ties the two together so a future ICON grid change
    # cannot silently pair new codes with stale coordinates.
    grid_id = f"icon-global-R03B07-za-{n}"
    doc["grid_id"] = grid_id
    with gzip.open(a.out, "wb", compresslevel=9) as f:
        f.write(json.dumps(doc, separators=(",", ":")).encode("utf-8"))

    grid_path = os.path.join(out_dir, "za-thunder-grid.json.gz")
    grid = {
        "schema": 1,
        "grid_id": grid_id,
        "licence": LICENCE,
        "source": SOURCE_URL,
        "cell_count": int(n),
        # int16 hundredths of a degree: ~1.1 km, well inside the 13 km spacing.
        "cell_lat_e2": b64(np.round(lat * 100).astype(np.int16)),
        "cell_lon_e2": b64(np.round(lon * 100).astype(np.int16)),
    }
    with gzip.open(grid_path, "wb", compresslevel=9) as f:
        f.write(json.dumps(grid, separators=(",", ":")).encode("utf-8"))

    log(
        f"wrote {a.out}: {os.path.getsize(a.out)/1024:.1f} KiB gz "
        f"({n} cells x {len(days)} days)"
    )
    log(
        f"wrote {grid_path}: {os.path.getsize(grid_path)/1024:.1f} KiB gz "
        f"(static, grid_id={grid_id})"
    )
    for d, ok, c in zip(days, assessed, codes):
        arr = np.frombuffer(base64.b64decode(c), dtype=np.uint8)
        t = int(np.isin(arr, THUNDER_CODES).sum())
        log(f"  {d}  {'assessed' if ok else 'UNKNOWN '}  thunder cells: {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
