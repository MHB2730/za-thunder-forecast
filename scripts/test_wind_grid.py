#!/usr/bin/env python3
"""Pin the wind-grid builder's encoding and fail-safe rules. No network.

    python scripts/test_wind_grid.py

Runs the real `fetch_dwd_wind.main()` with the download, decode and cell index
replaced by in-memory stubs, so the whole path from field to file is exercised:
  * quantisation round-trip (to within half a step) and clamping at ±63.5 m/s
  * the written file's schema keys, and that it decodes back to the input
  * a grid mismatch (published thunder grid, or a field of the wrong size)
    stops the build and writes nothing
  * too few decoded steps exits non-zero and writes nothing
  * a step missing u OR v is left out of `steps`, never zeroed
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import shutil
import sys
import tempfile
import urllib.error

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_dwd_thunder as thunder  # noqa: E402
import fetch_dwd_wind as wind  # noqa: E402

failures = 0


def check(name, cond, got=None):
    global failures
    if cond:
        return
    failures += 1
    print(f"FAIL  {name}" + ("" if got is None else f"\n      got: {got!r}"))


# ── Quantisation ────────────────────────────────────────────────────────────
x = np.array([0.0, 0.24, 0.26, -0.26, 1.0, -7.3, 12.74, 63.5, -63.5])
back = wind.dequantise(wind.quantise(x))
check("round-trip within half a step", np.all(np.abs(back - x) <= 0.25 + 1e-6),
      list(zip(x, back)))
check("quantise gives int8", wind.quantise(x).dtype == np.int8)
check("0.5 m/s steps", wind.quantise(np.array([1.0, -1.5])).tolist() == [2, -3])
big = wind.quantise(np.array([63.6, 80.0, 1e6, -63.6, -80.0, -1e6]))
check("clamped to ±127", big.tolist() == [127, 127, 127, -127, -127, -127], big.tolist())
check("clamp is 63.5 m/s", wind.CLAMP == 63.5 and wind.SCALE == 0.5)
check("-128 never produced", int(wind.quantise(np.array([-1e9]))[0]) == -127)

# ── Stubbed pipeline ────────────────────────────────────────────────────────
N = 50
G = wind.ICON_GLOBAL_VALUES
IDX = np.arange(0, N * 7, 7, dtype=np.int32)  # scattered cells in the global grid
RUN = "2026111106"
rng = np.random.default_rng(1)
TRUTH = {(s, c): rng.uniform(-20, 20, N) for s in range(31) for c in "uv"}

state = {"missing": set(), "size": G}


def fake_cell_index(run, cache):
    return IDX, np.zeros(N), np.zeros(N)


def fake_fetch(url):
    name = url.rsplit("/", 1)[1]
    step = int(name.split("_")[-3])
    comp = "u" if name.endswith("_U_10M.grib2.bz2") else "v"
    if (step, comp) in state["missing"]:
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
    return json.dumps([step, comp]).encode()


def fake_grib_values(raw):
    step, comp = json.loads(raw)
    full = np.zeros(state["size"])
    full[IDX[IDX < state["size"]]] = TRUTH[(step, comp)][: (IDX < state["size"]).sum()]
    return full


thunder.cell_index = fake_cell_index
thunder.fetch = fake_fetch
thunder.grib_values = fake_grib_values

tmp = tempfile.mkdtemp(prefix="wind-test-")


def run(*extra, missing=(), size=G, grid=None):
    state["missing"] = set(missing)
    state["size"] = size
    out = os.path.join(tmp, "out", "za-wind.json.gz")
    if os.path.exists(out):
        os.remove(out)
    gpath = os.path.join(tmp, "out", "za-thunder-grid.json.gz")
    if os.path.exists(gpath):
        os.remove(gpath)
    if grid is not None:
        os.makedirs(os.path.dirname(gpath), exist_ok=True)
        with gzip.open(gpath, "wb") as f:
            f.write(json.dumps(grid).encode())
    rc = wind.main(["--run", RUN, "--out", out, "--cache", tmp, *extra])
    return rc, out


try:
    # ── A good run: schema and content ──────────────────────────────────────
    good_grid = {"grid_id": f"icon-global-R03B07-za-{N}", "cell_count": N}
    rc, out = run(grid=good_grid)
    check("good run exits 0", rc == 0, rc)
    doc = json.loads(gzip.open(out).read())
    keys = {"schema", "model", "licence", "source", "run", "generated", "grid_id",
            "cell_count", "unit", "scale", "clamp", "steps_requested",
            "steps_decoded", "steps", "valid", "u", "v"}
    check("schema keys exact", set(doc) == keys, sorted(set(doc) ^ keys))
    check("schema 1", doc["schema"] == 1)
    check("grid_id matches thunder grid", doc["grid_id"] == good_grid["grid_id"])
    check("cell_count", doc["cell_count"] == N)
    check("steps 0..30", doc["steps"] == list(range(31)), doc["steps"])
    check("valid times hourly from run",
          doc["valid"][0] == "2026-11-11T06:00:00+00:00"
          and doc["valid"][30] == "2026-11-12T12:00:00+00:00", doc["valid"][::30])
    check("run ISO", doc["run"] == "2026-11-11T06:00:00+00:00", doc["run"])
    check("licence carried", doc["licence"] == thunder.LICENCE)
    check("source carried", doc["source"] == thunder.SOURCE_URL)
    check("scale 0.5", doc["scale"] == 0.5)
    check("one u and v per step", len(doc["u"]) == len(doc["v"]) == 31)
    for s in (0, 17, 30):
        for c in "uv":
            q = np.frombuffer(base64.b64decode(doc[c][doc["steps"].index(s)]), dtype=np.int8)
            check(f"{c} +{s}h is cell_count bytes", q.size == N, q.size)
            check(f"{c} +{s}h decodes to the field",
                  np.all(np.abs(q * doc["scale"] - TRUTH[(s, c)]) <= 0.25 + 1e-6))

    # ── A step missing one component is left out, not zeroed ────────────────
    rc, out = run(missing={(5, "v"), (6, "u")})
    doc = json.loads(gzip.open(out).read())
    check("partial run still publishes", rc == 0, rc)
    check("missing steps absent", 5 not in doc["steps"] and 6 not in doc["steps"],
          doc["steps"])
    check("valid follows steps", len(doc["valid"]) == len(doc["steps"]) == 29)
    check("steps_decoded honest", doc["steps_decoded"] == 29 and doc["steps_requested"] == 31)

    # ── Too few steps: non-zero, nothing written ────────────────────────────
    rc, out = run(missing={(s, "u") for s in range(7)})  # 24 of 31 < ceil(0.8*31)=25
    check("too few steps exits non-zero", rc != 0, rc)
    check("too few steps writes nothing", not os.path.exists(out))
    rc, out = run("--min-steps", "31", missing={(30, "v")})
    check("--min-steps honoured", rc != 0 and not os.path.exists(out), rc)

    # ── Grid mismatch: refused, nothing written ─────────────────────────────
    rc, out = run(grid={"grid_id": "icon-global-R03B07-za-13024", "cell_count": 13024})
    check("published grid mismatch refused", rc != 0, rc)
    check("grid mismatch writes nothing", not os.path.exists(out))
    rc, out = run(grid={"grid_id": f"icon-global-R03B07-za-{N}", "cell_count": N + 1})
    check("cell_count mismatch refused", rc != 0 and not os.path.exists(out), rc)
    rc, out = run(size=G - 1)
    check("wrong-size field refused", rc != 0, rc)
    check("wrong-size field writes nothing", not os.path.exists(out))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(("FAILED  " if failures else "OK  ") +
      (f"{failures} failure(s)" if failures else "all wind-grid cases pass"))
sys.exit(1 if failures else 0)
