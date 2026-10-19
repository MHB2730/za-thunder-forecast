#!/usr/bin/env python3
"""Tests for fetch_gfs_mountain.py. Offline: decode is replaced by a synthetic field.

    python scripts/test_mountain.py
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_gfs_mountain as M  # noqa: E402

FAILS = 0


def check(name, ok):
    global FAILS
    print(("  ok   " if ok else "  FAIL ") + name)
    if not ok:
        FAILS += 1


def grid():
    s, n, w, e = M.BOX
    lat = np.arange(s, n + 1e-9, M.RES)
    lon = np.arange(w, e + 1e-9, M.RES)
    return lat, lon


def synthetic(step: int, *, t_shift=0.0, tp=None) -> dict:
    lat, lon = grid()
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    d = {"lat": lat, "lon": lon}
    for k, p in enumerate(M.LEVELS):
        d[("gh", p)] = M.STD_H[k] + 10.0 * (LO - 25)          # varies across the box
        d[("t", p)] = 288.15 - 0.0065 * M.STD_H[k] + t_shift + 0 * LA
        d[("r", p)] = 50 + 0 * LA
        d[("u", p)] = 5 + 0 * LA
        d[("v", p)] = -3 + 0 * LA
    d["orog"] = 1000 + 0 * LA
    d["t2"] = 290.0 + 0 * LA
    d["frz"] = 3500 + 0 * LA
    if step > 0:
        d["tp"] = (tp if tp is not None else 1.0) + 0 * LA
    return d


# ── rain buckets ──────────────────────────────────────────────────────────────
print("rain_between")
acc = {3: np.array([3.0]), 6: np.array([5.0]), 9: np.array([2.0]), 12: np.array([2.0]), 126: np.array([12.0])}
check("first step of a block: its value is the amount (3 mm / 3 h = 1 mm/h)",
      np.allclose(M.rain_between(0, 3, acc), 1.0))
check("same block: difference (5-3 = 2 mm / 3 h)", np.allclose(M.rain_between(3, 6, acc), 2 / 3))
check("new block at 6: f009 carries 6-9 (2 mm / 3 h)", np.allclose(M.rain_between(6, 9, acc), 2 / 3))
check("f012 shares the 6-12 block with f009: 2-2 = 0", np.allclose(M.rain_between(9, 12, acc), 0.0))
check("6-hourly past +120: f126 carries 120-126 (12 mm / 6 h)", np.allclose(M.rain_between(120, 126, acc), 2.0))
check("missing previous step in the same block: unknown", M.rain_between(3, 6, {6: np.array([5.0])}) is None)
check("a decode glitch never gives negative rain",
      np.allclose(M.rain_between(3, 6, {3: np.array([5.0]), 6: np.array([4.0])}), 0.0))
check("a gap before the block start is unknown (prev 3, step 12)", M.rain_between(3, 12, acc) is None)

# ── encodings ─────────────────────────────────────────────────────────────────
print("encodings")
check("NaN is UNKNOWN in every uint8 plane", M.enc_t(np.array([np.nan]))[0] == 255)
check("temperature -60..+67 C round-trips to 0.5 C",
      abs((M.enc_t(np.array([-12.3]))[0] * 0.5 - 60) - (-12.3)) <= 0.25)
check("a real value can never collide with UNKNOWN (caps at 254)", M.enc_t(np.array([200.0]))[0] == 254)
check("rain caps at 25.4 mm/h, never 255", M.enc_rain(np.array([99.0]))[0] == 254)
check("0 C level in 25 m units", M.enc_frz(np.array([3500.0]))[0] == 140)
check("height offset from the standard level, 8 m units",
      M.enc_z(np.array([M.STD_H[6] + 80.0]), 6)[0] == 138)
check("std heights rise with falling pressure", all(np.diff(M.STD_H) > 0))

# ── sanity stop ───────────────────────────────────────────────────────────────
print("sanity stop")
bad = synthetic(3); bad[("t", 700)] = bad[("t", 700)] + 200
try:
    M.sanity(bad, 3); check("absurd temperature stops the run", False)
except M.Insane:
    check("absurd temperature stops the run", True)
bad = synthetic(3); bad[("gh", 700)] = bad[("gh", 750)] - 10
try:
    M.sanity(bad, 3); check("heights that do not rise stop the run", False)
except M.Insane:
    check("heights that do not rise stop the run", True)
try:
    M.sanity(synthetic(3), 3); check("a sane step passes", True)
except M.Insane:
    check("a sane step passes", False)

# ── tiles: shared edges ───────────────────────────────────────────────────────
print("tiles")
dec = {0: synthetic(0), 3: synthetic(3)}
rain = {0: None, 3: M.rain_between(0, 3, {3: dec[3]["tp"]})}
import datetime as dt  # noqa: E402
run = dt.datetime(2026, 10, 19, 0)
a = M.build_tile("s30e028", -30.0, 28.0, [0, 3], dec, rain, run, "x")
b = M.build_tile("s30e030", -30.0, 30.0, [0, 3], dec, rain, run, "x")


def plane(t, k, shape):
    raw = base64.b64decode(t[k])
    return np.frombuffer(raw, np.uint8).reshape(shape)


za = plane(a, "z", (2, len(M.LEVELS), 9, 9)); zb = plane(b, "z", (2, len(M.LEVELS), 9, 9))
check("east edge of one tile == west edge of the next", np.array_equal(za[..., -1], zb[..., 0]))
ra = plane(a, "rain", (2, 9, 9))
check("step 0 has no rain reading: UNKNOWN, not dry", np.all(ra[0] == 255))
check("step 3 rain encoded (1 mm/3 h -> 0.33 mm/h -> 3)", np.all(ra[1] == 3))
orog = np.frombuffer(base64.b64decode(a["orog"]), "<u2").reshape(9, 9)
check("model ground in metres", np.all(orog == 1000))
check("file says its own encodings", a["enc"]["unknown"] == 255 and a["levels"] == M.LEVELS)

# ── main: fail-safe ───────────────────────────────────────────────────────────
print("main")
real_decode = M.decode


def run_main(missing: set[int]):
    out = tempfile.mkdtemp()
    tiles = os.path.join(out, "tiles.json")
    json.dump({"tiles": ["s30e028"]}, open(tiles, "w"))
    M.decode = lambda raw, s: None if s in missing else synthetic(s)
    try:
        code = M.main(["--out", os.path.join(out, "v1"), "--tiles", tiles, "--run", "2026101900",
                       "--steps", ",".join(str(s) for s in range(0, 31, 3))], fetch=lambda url: b"GRIB")
    finally:
        M.decode = real_decode
    return code, os.path.join(out, "v1")


code, d = run_main(missing={6, 9, 12})            # 8 of 11 = 73 % < 80 %
check("below 80 % of steps: exit 1", code == 1)
check("below 80 % of steps: NOTHING written", not os.path.exists(os.path.join(d, "s30e028.json.gz")))
code, d = run_main(missing={6})
check("one step missing: still published", code == 0)
t = json.loads(gzip.decompress(open(os.path.join(d, "s30e028.json.gz"), "rb").read()))
check("the missing step is flagged not ok", t["ok"][2] is False and all(t["ok"][i] for i in (0, 1, 3)))
tp = plane(t, "t", (11, len(M.LEVELS), 9, 9))
check("the missing step is all UNKNOWN", np.all(tp[2] == 255))
rr = plane(t, "rain", (11, 9, 9))
check("f009 opens a new 6-h block, so its rain is known even with f006 missing", np.all(rr[3] != 255))
code, d = run_main(missing={3})
t = json.loads(gzip.decompress(open(os.path.join(d, "s30e028.json.gz"), "rb").read()))
rr = plane(t, "rain", (11, 9, 9))
check("f006 shares a block with the missing f003: its rain is UNKNOWN, not dry", np.all(rr[2] == 255))
check("index lists the tile and the run", json.load(open(os.path.join(d, "index.json")))["tiles"] == ["s30e028"])

print(f"\n{FAILS} failure(s)")
sys.exit(1 if FAILS else 0)
