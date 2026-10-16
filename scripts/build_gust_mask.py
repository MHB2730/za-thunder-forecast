#!/usr/bin/env python3
"""Build data/za-gust-mask.json — the ICON cells the gust feed publishes.

    python scripts/build_gust_mask.py <za_boundary.json> [--cache .dwd-cache]

The thunder grid is the whole South Africa BOX: 13,024 cells, of which about
45 % are sea, Namibia, Botswana, Zimbabwe or Mozambique. Thunder costs nothing
there (its planes are almost all one value). Gusts are different: every cell
carries a real, varying number, and publishing the box made the file 520 KiB
gzipped — too heavy for a phone four times a day. Keeping only cells within
BUFFER_KM of South Africa (Lesotho included: the boundary is the outer ring)
leaves 8,118 cells and roughly a third of the size, with no loss of precision.
Cells outside carry the UNKNOWN sentinel, so the app says "no gust forecast"
there rather than calm.

`za_boundary.json` is the outer ring the app ships
(trailtether_app/assets/map/za_boundary.json). Only the derived cell mask is
committed here, not the boundary. Re-run this if ICON's grid changes (the
`grid_id` check in fetch_dwd_gust.py refuses a stale mask).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial import KM_PER_DEG  # noqa: E402

BUFFER_KM = 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("boundary")
    ap.add_argument("--cache", default=".dwd-cache")
    ap.add_argument("--out", default="data/za-gust-mask.json")
    a = ap.parse_args()

    z = np.load(os.path.join(a.cache, "za_cells.npz"))
    lat, lon = z["lat"], z["lon"]
    ring = np.array(json.load(open(a.boundary, encoding="utf-8"))["coordinates"][0])

    inside = np.zeros(lat.size, dtype=bool)
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        if y1 == y2:
            continue
        crosses = (y1 > lat) != (y2 > lat)
        x_at = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
        inside ^= crosses & (lon < x_at)

    near = np.full(lat.size, np.inf)
    for vx, vy in ring:
        dy = (lat - vy) * KM_PER_DEG
        dx = (lon - vx) * KM_PER_DEG * np.cos(np.radians(vy))
        near = np.minimum(near, np.hypot(dx, dy))

    keep = inside | (near <= BUFFER_KM)
    doc = {
        "grid_id": f"icon-global-R03B07-za-{lat.size}",
        "buffer_km": BUFFER_KM,
        "built_from": "trailtether_app/assets/map/za_boundary.json (outer ring)",
        "cell_count": int(lat.size),
        "kept": int(keep.sum()),
        # np.packbits, big-endian bit order: cell i is bit (7 - i % 8) of byte i // 8.
        "keep_bits": base64.b64encode(np.packbits(keep).tobytes()).decode("ascii"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {a.out}: {doc['kept']} of {doc['cell_count']} cells "
          f"(inside {int(inside.sum())}, within {BUFFER_KM:g} km {doc['kept']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
