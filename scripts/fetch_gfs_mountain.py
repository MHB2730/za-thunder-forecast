#!/usr/bin/env python3
"""Mountain weather for TrailTether: NOAA GFS pressure levels over South Africa,
cut into 2 x 2 degree area files the phone caches offline.

    python scripts/fetch_gfs_mountain.py --out out/mtn/v1
    python scripts/fetch_gfs_mountain.py --out out/mtn/v1 --steps 0,3,6 --run 2026101900   # dev

Plan: trailtether `SCOPE-3d-weather.md`. Evidence: trailtether
`design/weather_3d_prototype_2026-10/` (six-week check, 1 Sept - 15 Oct 2026).

WHY THIS EXISTS
---------------
GFS smooths the Drakensberg: it puts Mafadi's ground at 2,395 m (the summit is
3,444 m), so its "ground" weather is for air a kilometre below the hiker. The
phone reads this file's 3D air (eleven pressure levels) at the REAL height of
the ground instead. Checked against six weeks of balloons, stations and
satellite: summit-height air ~0.8 C, the 0 C level ~144 m, mean wind ~8 km/h.
Cloud on the summit did NOT check out (satellite clear ~2 in 3 times GFS said
cloud); RH is still shipped because the owner asked to see cloud tops, and the
app must label every cloud reading a guess. Do not add a cloud verdict here.

WHAT IS IN A FILE (schema 1)
----------------------------
One file per tile, named by its south-west corner (`s30e028` = 30-28 S,
28-30 E), 9 x 9 points at 0.25 deg with EDGES SHARED between neighbours so the
phone's interpolation never falls off a tile. Every plane is base64 uint8 (orog
uint16 LE), rows south to north, columns west to east, and 255 means UNKNOWN in
every uint8 plane (65535 in orog). A step that did not decode is all-unknown
and `ok[i]` is false: the phone must treat it as unknown, never as dry or calm.

    level planes  [step][level][row][col]   z t rh u v
    surface       [step][row][col]          t2 frz rain
    static        [row][col]                orog

Encodings (also written into the file as `enc`, which the phone reads):
    z     metres above the level's standard-atmosphere height: (gh - std)/8 + 128
    t, t2 deg C: (T + 60) * 2
    rh    percent * 2.5
    u, v  m/s + 127
    frz   0 C level, metres / 25           (0 .. 6,350 m)
    rain  mean rate over the time since the previous step, mm/h * 10 (cap 25.4)

FAIL-SAFE (the contract the thunder and gust feeds already keep)
----------------------------------------------------------------
* A step counts only if EVERY field decoded. A missing step stays unknown.
* Fewer than MIN_SHARE of the steps -> exit 1 and write nothing. The last good
  files stay on R2; the phone ages them out after 30 h.
* A sanity stop checks every decoded value against physical limits BEFORE
  anything is written: a run that finished is not a run that is right.

Licence: NOAA/NCEP GFS is US-government work, public domain (17 U.S.C. 105).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import json
import os
import sys
import tempfile
import time

import numpy as np
import eccodes as ec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_gfs import (  # noqa: E402
    CYCLES, LICENCE, MODEL, SOURCE_URL, UA, _fetch, _published, block_start,
)
from runs import pick_run  # noqa: E402

FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
SCHEMA = 1
LEVELS = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500]
# 3-hourly to +120 h, then 6-hourly to +168 h (49 steps). Every 6-hourly step
# lands on a GFS accumulation boundary, which `rain_between` relies on.
STEPS = list(range(0, 121, 3)) + list(range(126, 169, 6))
BOX = (-36.0, -22.0, 16.0, 34.0)          # south, north, west, east
TILE_DEG = 2.0
RES = 0.25
N = int(TILE_DEG / RES) + 1               # 9 points a side, edges shared
MIN_SHARE = 0.8
UNKNOWN = 255
OROG_UNKNOWN = 65535
SLEEP_S = 1.2                             # NOMADS is somebody else's free service


def std_height(p_hpa: float) -> float:
    """ICAO standard-atmosphere height of a pressure level, metres."""
    return 44330.8 * (1.0 - (p_hpa / 1013.25) ** 0.190263)


STD_H = [round(std_height(p)) for p in LEVELS]


# ── encodings (the phone mirrors these exactly; see test_mountain.py) ─────────

def _u8(x: np.ndarray) -> np.ndarray:
    out = np.where(np.isfinite(x), np.clip(np.rint(x), 0, 254), UNKNOWN)
    return out.astype(np.uint8)


def enc_z(gh: np.ndarray, level_idx: int) -> np.ndarray:
    return _u8((gh - STD_H[level_idx]) / 8.0 + 128.0)


def enc_t(t_c: np.ndarray) -> np.ndarray:
    return _u8((t_c + 60.0) * 2.0)


def enc_rh(rh: np.ndarray) -> np.ndarray:
    return _u8(rh * 2.5)


def enc_wind(ms: np.ndarray) -> np.ndarray:
    return _u8(ms + 127.0)


def enc_frz(m: np.ndarray) -> np.ndarray:
    return _u8(m / 25.0)


def enc_rain(mm_h: np.ndarray) -> np.ndarray:
    return _u8(np.minimum(mm_h * 10.0, 254.0))


ENC = {
    "z": {"scale": 8.0, "offset": 128, "relative_to": "std_h"},
    "t": {"scale": 0.5, "offset": -60.0},
    "t2": {"scale": 0.5, "offset": -60.0},
    "rh": {"scale": 0.4, "offset": 0.0},
    "u": {"scale": 1.0, "offset": -127.0},
    "v": {"scale": 1.0, "offset": -127.0},
    "frz": {"scale": 25.0, "offset": 0.0},
    "rain": {"scale": 0.1, "offset": 0.0, "unit": "mm/h"},
    "unknown": UNKNOWN,
    "orog_unknown": OROG_UNKNOWN,
}


# ── rain: GFS accumulates in 6-hour blocks ────────────────────────────────────

def rain_between(prev: int, step: int, acc: dict[int, np.ndarray]) -> np.ndarray | None:
    """Mean rate (mm/h) of the rain that fell between `prev` and `step`.

    `acc[s]` is the since-block-start accumulation GFS reports at step `s`
    (blocks reset at 0/6/12/18 h; a step ON a boundary closes the block that
    ends there, see `model_gfs.block_start`). When `step`'s block starts at or
    after `prev`, its value IS the amount since `prev` (3-hourly steps in the
    first half of a block, and every 6-hourly step past +120 h). Otherwise the
    two steps share a block and the amount is the difference.
    """
    cur = acc.get(step)
    if cur is None or step <= prev:
        return None
    b = block_start(step)
    if b >= prev:
        if b > prev:
            return None   # a gap: rain before the block start was not seen
        amount = cur
    else:
        before = acc.get(prev)
        if before is None:
            return None
        # Monotonic within a block; a decode glitch must not produce negative
        # rain, which would MASK a wet spell.
        amount = np.maximum(0.0, cur - before)
    return amount / float(step - prev)


# ── fetch + decode ────────────────────────────────────────────────────────────

def step_url(run: dt.datetime, step: int) -> str:
    s, n, w, e = BOX
    lev = "".join(f"&lev_{p}_mb=on" for p in LEVELS)
    return (
        f"{FILTER}?file=gfs.t{run:%H}z.pgrb2.0p25.f{step:03d}"
        "&var_HGT=on&var_TMP=on&var_RH=on&var_UGRD=on&var_VGRD=on&var_APCP=on"
        f"{lev}&lev_2_m_above_ground=on&lev_surface=on&lev_0C_isotherm=on"
        f"&subregion=&leftlon={w}&rightlon={e}&toplat={n}&bottomlat={s}"
        f"&dir=%2Fgfs.{run:%Y%m%d}%2F{run:%H}%2Fatmos"
    )


def decode(raw: bytes, step: int) -> dict | None:
    """Every field of one step on the BOX grid (lat ascending), or None if any is missing."""
    got: dict = {}
    lat = lon = None
    fd, path = tempfile.mkstemp(suffix=".grib2")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        with open(path, "rb") as f:
            while True:
                g = ec.codes_grib_new_from_file(f)
                if g is None:
                    break
                try:
                    name = ec.codes_get(g, "shortName")
                    tl = ec.codes_get(g, "typeOfLevel")
                    lv = ec.codes_get(g, "level")
                    ni, nj = ec.codes_get(g, "Ni"), ec.codes_get(g, "Nj")
                    v = ec.codes_get_values(g).reshape(nj, ni).astype(np.float64)
                    la = ec.codes_get_array(g, "distinctLatitudes")
                    lo = ec.codes_get_array(g, "distinctLongitudes")
                    if la[0] > la[-1]:
                        la, v = la[::-1], v[::-1]
                    lat, lon = la, lo
                    if tl == "isobaricInhPa" and name in ("gh", "t", "r", "u", "v") and lv in LEVELS:
                        got[(name, lv)] = v
                    elif name == "orog" and tl == "surface":
                        got["orog"] = v
                    elif name == "2t" and tl == "heightAboveGround":
                        got["t2"] = v
                    elif name == "gh" and tl == "isothermZero":
                        got["frz"] = v
                    elif name == "tp" and tl == "surface":
                        # NOMADS sends tp twice (since block start, since run):
                        # keep the since-block-start window.
                        rng = ec.codes_get(g, "stepRange")
                        if rng == f"{block_start(step)}-{step}":
                            got["tp"] = v
                finally:
                    ec.codes_release(g)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    need = [(n_, p) for p in LEVELS for n_ in ("gh", "t", "r", "u", "v")] + ["orog", "t2", "frz"]
    if any(k not in got for k in need):
        return None
    if step > 0 and "tp" not in got:
        return None
    got["lat"], got["lon"] = lat, lon
    return got


# ── sanity stop ───────────────────────────────────────────────────────────────

class Insane(RuntimeError):
    pass


def sanity(d: dict, step: int) -> None:
    """Physical limits. Raises Insane: the whole run is then written NOWHERE."""
    def rng(x, lo, hi, what):
        x = np.asarray(x)
        if not np.all(np.isfinite(x)) or x.min() < lo or x.max() > hi:
            raise Insane(f"f{step:03d} {what} out of range [{lo}, {hi}]: {np.nanmin(x):.1f}..{np.nanmax(x):.1f}")
    for p in LEVELS:
        rng(d[("t", p)] - 273.15, -90, 50, f"t{p}")
        rng(d[("r", p)], 0, 105, f"rh{p}")
        rng(np.hypot(d[("u", p)], d[("v", p)]) * 3.6, 0, 400, f"wind{p}")
    rng(d["t2"] - 273.15, -40, 50, "t2m")
    rng(d["frz"], -500, 7000, "0C level")
    rng(d["orog"], -100, 4000, "model ground")
    z = np.stack([d[("gh", p)] for p in LEVELS])
    if np.any(np.diff(z, axis=0) <= 0):
        raise Insane(f"f{step:03d} pressure-level heights do not rise")
    if "tp" in d:
        rng(d["tp"], 0, 400, "6 h rain")


# ── tiles ─────────────────────────────────────────────────────────────────────

def tile_name(lat0: float, lon0: float) -> str:
    return f"s{int(round(-lat0)):02d}e{int(round(lon0)):03d}"


def tile_index(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    i0 = int(round((lat0 - lat[0]) / RES)); j0 = int(round((lon0 - lon[0]) / RES))
    if i0 < 0 or j0 < 0 or i0 + N > len(lat) or j0 + N > len(lon):
        raise ValueError(f"tile {tile_name(lat0, lon0)} outside the decoded grid")
    if abs(lat[i0] - lat0) > 1e-6 or abs(lon[j0] - lon0) > 1e-6:
        raise ValueError(f"tile {tile_name(lat0, lon0)} is not on the 0.25 deg grid")
    return slice(i0, i0 + N), slice(j0, j0 + N)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def build_tile(name: str, lat0: float, lon0: float, steps: list[int], decoded: dict[int, dict],
               rain: dict[int, np.ndarray | None], run: dt.datetime, made: str) -> dict:
    any_d = next(iter(decoded.values()))
    si, sj = tile_index(any_d["lat"], any_d["lon"], lat0, lon0)
    ns, nl = len(steps), len(LEVELS)
    planes = {k: np.full((ns, nl, N, N), UNKNOWN, np.uint8) for k in ("z", "t", "rh", "u", "v")}
    surf = {k: np.full((ns, N, N), UNKNOWN, np.uint8) for k in ("t2", "frz", "rain")}
    ok = []
    for i, s in enumerate(steps):
        d = decoded.get(s)
        ok.append(d is not None)
        if d is None:
            continue
        for k, p in enumerate(LEVELS):
            planes["z"][i, k] = enc_z(d[("gh", p)][si, sj], k)
            planes["t"][i, k] = enc_t(d[("t", p)][si, sj] - 273.15)
            planes["rh"][i, k] = enc_rh(d[("r", p)][si, sj])
            planes["u"][i, k] = enc_wind(d[("u", p)][si, sj])
            planes["v"][i, k] = enc_wind(d[("v", p)][si, sj])
        surf["t2"][i] = enc_t(d["t2"][si, sj] - 273.15)
        surf["frz"][i] = enc_frz(d["frz"][si, sj])
        r = rain.get(s)
        if r is not None:
            surf["rain"][i] = enc_rain(r[si, sj])
    orog = np.where(np.isfinite(any_d["orog"][si, sj]),
                    np.clip(np.rint(any_d["orog"][si, sj]), 0, 65534), OROG_UNKNOWN).astype("<u2")
    return {
        "schema": SCHEMA, "model": MODEL, "licence": LICENCE, "source": SOURCE_URL,
        "run": run.strftime("%Y-%m-%dT%H:00:00Z"), "made": made,
        "tile": name, "lat0": lat0, "lon0": lon0, "n": N, "res": RES,
        "levels": LEVELS, "std_h": STD_H, "steps": steps, "ok": ok, "enc": ENC,
        "orog": b64(orog),
        **{k: b64(v) for k, v in planes.items()},
        **{k: b64(v) for k, v in surf.items()},
    }


def _utcnow() -> dt.datetime:
    """Naive UTC, which runs.pick_run expects."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def load_tiles(path: str) -> list[tuple[float, float]]:
    with open(path, encoding="utf-8") as f:
        names = json.load(f)["tiles"]
    out = []
    for nm in names:
        out.append((-float(nm[1:3]), float(nm[4:7])))
    return out


def write_json_gz(path: str, obj: dict) -> int:
    raw = json.dumps(obj, separators=(",", ":")).encode()
    data = gzip.compress(raw, 9)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def main(argv: list[str] | None = None, fetch=_fetch, now: dt.datetime | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out/mtn/v1")
    ap.add_argument("--tiles", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mtn-tiles.json"))
    ap.add_argument("--run", help="YYYYMMDDHH (default: newest fully published)")
    ap.add_argument("--steps", help="comma list (dev only; the published ladder is STEPS)")
    a = ap.parse_args(argv)

    steps = [int(x) for x in a.steps.split(",")] if a.steps else STEPS
    if a.run:
        run = dt.datetime.strptime(a.run, "%Y%m%d%H")
    else:
        run = pick_run(now or _utcnow(), CYCLES, lambda r: _published(r, steps[-1]))
    print(f"GFS run {run:%Y-%m-%d %HZ}, {len(steps)} steps", flush=True)

    decoded: dict[int, dict] = {}
    for s in steps:
        try:
            d = decode(fetch(step_url(run, s)), s)
        except Exception as e:  # noqa: BLE001 - one bad step is absent, not fatal
            print(f"  f{s:03d}: {type(e).__name__}: {e}", flush=True)
            d = None
        if d is None:
            print(f"  f{s:03d}: incomplete, left UNKNOWN", flush=True)
        else:
            sanity(d, s)            # Insane propagates: nothing is written
            decoded[s] = d
        if fetch is _fetch:
            time.sleep(SLEEP_S)

    share = len(decoded) / len(steps)
    print(f"decoded {len(decoded)}/{len(steps)} steps ({share:.0%})", flush=True)
    if share < MIN_SHARE:
        print(f"below {MIN_SHARE:.0%}: writing NOTHING; the last good files stay", flush=True)
        return 1

    acc = {s: d["tp"] for s, d in decoded.items() if "tp" in d}
    rain: dict[int, np.ndarray | None] = {}
    for i, s in enumerate(steps):
        rain[s] = rain_between(steps[i - 1], s, acc) if i > 0 else None

    tiles = load_tiles(a.tiles)
    os.makedirs(a.out, exist_ok=True)
    made = _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    total = 0
    names = []
    for lat0, lon0 in tiles:
        nm = tile_name(lat0, lon0)
        total += write_json_gz(os.path.join(a.out, f"{nm}.json.gz"),
                               build_tile(nm, lat0, lon0, steps, decoded, rain, run, made))
        names.append(nm)
    index = {"schema": SCHEMA, "model": MODEL, "licence": LICENCE, "source": SOURCE_URL,
             "run": run.strftime("%Y-%m-%dT%H:00:00Z"), "made": made, "steps": steps,
             "ok": [s in decoded for s in steps], "tiles": names, "tile_deg": TILE_DEG}
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    print(f"wrote {len(names)} tiles, {total / 1024:.0f} KiB gzipped", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Insane as e:
        print(f"SANITY STOP: {e}. Nothing written.", flush=True)
        sys.exit(2)
