# GAUNTLET web prototype

The working prototype for the Mastercard Innovation Challenge 2026 submission. Six views, one
per judged criterion:

| View | Judged criterion | What it shows |
|---|---|---|
| **Home** | | The pitch, headline numbers, the arms-race curve, and the result worth reading twice |
| **Dashboard** | Efficacy | KPI strip, extraction curve with all twelve defender refits, attacker strategy stream, expert weights, risk bands to mitigation, and the five-way detector comparison with a derived F1 |
| **Simulator** | Fidelity | Calibration as ratios against the noise floor of real data split against itself, the eight rule trigger rates, hard-negative composition, fan-out variance to mean, fitted amount and category distributions, and the blindness rule |
| **The Loop** | Novelty | The instrumented curve with selectable refit markers, the entropy spike, how the strategy mutated, the final policy at full length, and victim selection |
| **Live** | Feasibility | Authorisations scoring against the real cost bands with a per-event explanation, plus what deploys, operating constraints, scalability, and commercial viability |
| **Demo** | Diversity | Attack explorer: pick one of eleven verticals, see its generative capability, its legal actions, and the chains the scripted and learned attackers actually produced |

## Run it

One command, whichever of these you already have installed.

```bash
cd submission/web

./run.sh                    # auto-detects Docker, then Node, then Python
docker compose up --build   # then open http://localhost:8080
open dist/index.html        # no toolchain at all, the build is self-contained
```

`./run.sh stop` shuts the container down. `./run.sh dev` starts the Vite dev server with hot
reload on port 5173.

## Why it is static

The prototype does not drive the Python pipeline. A full run is six stages and 3819 seconds
(63.7 minutes) on a GPU server, so a judge cannot press "run" and watch. Instead every number
on the page is transcribed from one real run's log by `tools/make_real_fixtures.py`, and the
raw log ships alongside it at `data/run.log` so the numbers can be audited.

## Where the numbers come from

```bash
python3 tools/make_real_fixtures.py
```

Reads `data/run.log` plus the committed calibration artifacts in `../artifacts/`, and writes
typed JSON into `src/data/`. It refuses to emit unless the run it parsed matches the one this
prototype reports, 150 co-adaptation updates, defender refits at updates 11 through 143, and
an extraction checksum of 945640.9, so a drifting parser fails loudly instead of quietly
publishing wrong figures.

Nothing is invented. Two things are labelled rather than presented as measured:

- **Precision, recall and F1 at the alert budget** are marked `derived`. `DetectionMetrics` in
  the Python does not compute F1, so it is computed here from `precision@budget` and the
  positive count, with the arithmetic stated on the page.
- **Any data file carrying a `_fixture` flag** triggers a page-wide warning banner. Real
  generated output carries no such flag, so the banner is normally absent. It exists because
  the earlier prototype branch shipped placeholder values, and a placeholder reaching a judge
  should be loud rather than silent.

The **Live** view samples individual authorisations rather than replaying measured ones. The
distribution parameters and the four band thresholds are fitted from real data, and the score is
a transparent linear stand-in whose weights are shown on screen, not the trained XGBoost
ensemble. The panel says so itself, and carries an `illustrative` badge and its seed.

The cost curve is labelled **relative cost**, never a currency figure: the simulator states
amounts are unit where event value is not to hand.

## Build

```bash
npm ci
npm run build     # tsc, then vite, then the CSP guard
```

The production build collapses to a single self-contained `dist/index.html` via
`vite-plugin-singlefile`, all JavaScript, CSS, data and fonts inlined, zero network requests.
That is why it can be opened straight off disk, and why the committed file and any hosted copy
are byte-identical.

`scripts/check-csp.mjs` runs as part of `npm run build` and fails it on any external reference,
`new Worker`, `new WebSocket`, or `XMLHttpRequest`, and on exceeding the size budget. The
failure mode it guards against is silent: a blocked request renders nothing and reports
nothing, so a broken page merely looks empty.

## Stack

React 18, Vite 6, TypeScript, Tailwind 4, visx for chart primitives, lucide-react for icons,
react-router-dom with hash routing. Fonts are self-hosted variable woff2, inlined as data URIs,
so the page looks the same offline as online.
