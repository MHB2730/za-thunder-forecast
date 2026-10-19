#!/usr/bin/env python3
"""Which 2 x 2 degree mountain-weather tiles to publish: every tile that touches
South Africa's national outer ring (which encloses Lesotho; eSwatini sits inside
tiles that already touch South Africa).

    python scripts/build_mtn_tiles.py --boundary <trailtether>/trailtether_app/assets/map/za_boundary.json

The boundary is the app's own asset and is not copied here; only the derived
tile list is committed (data/mtn-tiles.json). Re-run only if the tile size or
the country outline changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_gfs_mountain import BOX, TILE_DEG, tile_name  # noqa: E402


def inside(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for many points."""
    x, y = ring[:, 0], ring[:, 1]
    res = np.zeros(px.shape, bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        cross = ((y[i] > py) != (y[j] > py)) & (px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        res ^= cross
        j = i
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mtn-tiles.json"))
    a = ap.parse_args()
    g = json.load(open(a.boundary, encoding="utf-8"))
    ring = np.array(g["coordinates"][0], float)
    s, n, w, e = BOX
    names = []
    for lat0 in np.arange(s, n, TILE_DEG):
        for lon0 in np.arange(w, e, TILE_DEG):
            gx, gy = np.meshgrid(np.linspace(lon0, lon0 + TILE_DEG, 41), np.linspace(lat0, lat0 + TILE_DEG, 41))
            if inside(gx.ravel(), gy.ravel(), ring).any():
                names.append(tile_name(float(lat0), float(lon0)))
    json.dump({"tile_deg": TILE_DEG, "source": "trailtether assets/map/za_boundary.json (national outer ring)",
               "tiles": names}, open(a.out, "w", encoding="utf-8"), indent=1)
    print(len(names), "tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
