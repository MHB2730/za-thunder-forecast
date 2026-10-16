# za-thunder-forecast

Builds a South African thunderstorm forecast from [DWD ICON](https://www.dwd.de/EN/ourservices/nwp_forecast_data/nwp_forecast_data.html)
open data, four times a day, and publishes it as a small static file that
[TrailTether](https://trailtether.app) caches offline.

Output: `https://tiles.hilltrek.co.za/weather/za-thunder.json.gz` (plus a
static `za-thunder-grid.json.gz` carrying the cell coordinates).

Since 16 Oct 2026 the same job also publishes **wind gusts**:
`https://tiles.hilltrek.co.za/weather/za-gust.json.gz`, on the same cells.

## Wind gusts (`scripts/fetch_dwd_gust.py`)

met.no publishes `wind_speed_of_gust` only inside its Nordic model. Probed on
2026-10-16: 0 of 90 hourly blocks at 13 South African points (Berg, Cape,
Cederberg, Swartberg, Otter, Amatola, Magaliesberg, Blyde, Durban) in both its
`complete` and `compact` products, and none at London, Berlin, Madrid or
Nairobi; Oslo had 53 of 84. So every gust check in TrailTether ran on mean wind
or never fired. On 19 Oct 2026 the global models agreed on 100–132 km/h gusts
at Sani Pass while met.no's mean there was 28 km/h.

What the file holds, and why:

* **The field.** `VMAX_10M` (`max_i10fg`), checked against the live files: the
  maximum gust over the **preceding one hour** at every step, including past
  +78 h where ICON global only outputs every third hour. So every hour to +78 h
  is fetched (sampling every third would miss two thirds of the gusts), and
  past +78 h the published figure is honestly an **"at least"**.
* **The shape.** Per SAST day: eight 3-hour blocks when every hour was read
  (`full: true`), otherwise one `max` plane that is a lower bound. `cover` and
  `expect` give the hours read and the hours possible per block.
* **An area figure.** Each cell is the median of all cells within 25 km (the
  ensemble's reduction, see `spatial.py`), so one windy cell on the escarpment
  cannot stand for the area. km/h, rounded up.
* **A mask.** Only cells within 30 km of South Africa (Lesotho included) carry
  values (`data/za-gust-mask.json`, built by `scripts/build_gust_mask.py`);
  the rest carry `unknown` (255). That keeps the file around 180 KiB instead
  of 520.
* **Fail-safe.** Below `--min-steps` (90 of 112) decoded hours the script
  exits non-zero and writes nothing; the app ages the last good file out to
  "no gust forecast" after 30 h. A day with nothing decoded has no planes and
  reads as unknown, never calm.

It runs in the same workflow, after thunder is published and independent of
it. Attribution: the same DWD notice, with the modification stated.

## Why this exists

TrailTether's hiking score has a thunderstorm veto — the app's own comment calls
lightning "the #1 mountain killer", because on an exposed Drakensberg ridge there
is nowhere to shelter. **In South Africa that veto was dead, and silently so.**

It keys on `weatherCode >= 95 || thunderProb > 15`, and met.no publishes neither
here. Probed live across 16 points on 2026-08-06: `probability_of_thunder`
appears only inside met.no's Nordic model (53–54 hourly blocks at each of eight
Nordic controls, **zero** at Cathedral Peak, Sani Top, Durban, Cape Town, Congo,
Manaus, Jakarta and Kampala), and its `*andthunder` symbols are derived from that
same field — so outside the Nordics no thunder symbol is emitted either, Miami in
peak August storm season included. Both halves of the test are permanently false
in the target region, and every Berg storm day scored as storm-free.

DWD ICON's `WW` field is "weather interpretation (WMO)" — the same code space the
app already uses, thunder at 95/96/99 — and is genuinely global. On 2026-08-06 it
put a WMO 95 thunderstorm 1.1 km from Cathedral Peak on 10 August, which the app
could not see at all.

## The fail-safe contract

This whole feature exists because an absent reading was being presented as a
confident all-clear. The ingest must not repeat that mistake, so:

* A day is marked **assessed** only if *every* forecast step covering it
  downloaded and decoded. Any gap and the day is emitted as **unknown**.
* `code = 0` means "the model forecast no thunder". `code = 255` means "we could
  not tell". They are different values on purpose, and the consumer must render
  them differently.
* If too few steps survive overall, the script **exits non-zero and writes
  nothing** rather than publishing a partial file that reads as a quiet
  region-wide all-clear. A stale-but-honest artifact beats a fresh lie.

Consumers enforce a 30-hour staleness bound and fall back to "unknown" past it —
never to "clear".

## Why it is public, and why it is separate from the app

GitHub Actions is metered on private repos (2,000 Linux minutes/month), and the
owning account runs a $0 spending budget with "Stop usage" enabled. On
2026-08-15 that allowance ran out and Actions stopped, taking CI and this
four-times-daily safety job down together — the wrong coupling for a feature
about lightning. Actions is free and unlimited on **public** repos, so living
here means the forecast can never again be starved by a CI allowance.

Nothing proprietary is here. The app, the website, the backend and the hiking
score all stay private. The artifact this produces is already world-readable,
because the app fetches it without authentication.

## Running it

```bash
pip install numpy eccodes            # plus the ecCodes C library (libeccodes0)
python scripts/fetch_dwd_thunder.py --out out/za-thunder.json.gz
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--run 2026081500` | build a specific ICON run instead of the latest |
| `--steps 0,3,6` | a short run, for testing |
| `--min-steps N` | refuse to publish below N decoded steps (default 40) |
| `--bbox s,w,n,e` | override the South Africa box |

`--bbox` exists for a specific reason: over South Africa in winter every cell is
legitimately `0`, so a run here proves only that the script *can* emit zeros —
which is indistinguishable from a broken decode. Point it at a convective region
(India in August, say) and the same code path must come back non-zero.

## Secrets

The publish step needs three repository secrets: `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`.

Triggers are `schedule` and `workflow_dispatch` only. There is deliberately no
`pull_request` trigger, so a fork can never run this workflow and can never
reach those secrets.

## Licence

The **code** in this repository is © Hilltrek (Pty) Ltd, all rights reserved.
Public visibility is not a grant of licence.

The **data** it fetches is Deutscher Wetterdienst (DWD) ICON, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution is not
optional and is carried inside the published artifact so a consumer cannot ship
the data without it:

> Thunderstorm forecast: Deutscher Wetterdienst (DWD), ICON — CC BY 4.0

We reduce DWD's global hourly `WW` field to a per-day South African summary, so
"modified" is part of the required notice.
