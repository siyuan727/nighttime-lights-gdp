# Productivity extension: does light measure productivity, or only size?

This answers the strong version of your thesis. The raw light-GDP correlation
is strong because bright places are big places. The real test of whether light
carries economic information, rather than a head count, is whether it tracks
output **per worker** (labor productivity) and output **per person**
(prosperity). This module runs that test properly.

## Where the files go

| File | Put it in |
|---|---|
| `productivity.py` | `src/` |
| `run_productivity.py` | repo root, next to `main.py` |
| `test_productivity.py` | repo root (optional, for verification) |

Then, after `main.py` has built the panel at least once:

```
python run_productivity.py
```

It fetches county employment from BEA (table **CAEMP25N**, total jobs) using the
BEA key already in `src/config.py`, caches it to `data/raw/`, builds the
productivity panel, runs everything, and writes tables and a figure to
`output/`. If the employment pull fails (no network), it still runs the
per-capita (prosperity) half off population, so you always get a result.

## What it does, and why it survives a referee

Two objections a sharp judge will raise, both handled on purpose:

1. **"Dividing by workers is not controlling for workers."** GDP per worker is
   just log GDP minus log employment, so forming the ratio forces a coefficient
   of exactly one on size instead of estimating it. So alongside the ratio, the
   module runs the honest control: log GDP on log light **with log employment
   on the right-hand side**, inside two-way fixed effects. The light
   coefficient there is its partial association with output holding size fixed,
   which is the productivity signal, with size estimated rather than assumed.

2. **"Ratios that share a denominator correlate mechanically."** GDP per worker
   and light-per-worker both divide by employment, which can manufacture
   correlation out of nothing. The module sidesteps this by using **mean
   radiance** (`ntl_mean`, brightness per unit area) as the light measure: it is
   not divided by employment, so there is no shared denominator. The literal
   light-per-worker ratio is reported too, but **flagged** as the contaminated
   one, for contrast only.

## What it outputs

- `productivity_correlation_collapse.csv` — correlation of log light with log
  GDP (size), log GDP per capita (prosperity), and log GDP per worker
  (productivity). The headline: a strong size correlation that collapses for
  both per-person measures. This is your cleanest slide.
- `productivity_regressions.csv` — for each outcome, the ladder: OLS bivariate,
  OLS with a size control, OLS with a size control plus state effects, and
  two-way (county + year) fixed effects, all with county-clustered SEs. Compare
  the light coefficient across outcomes; it stays large only for total GDP.
- `productivity_horse_race.csv` — out-of-sample (GroupKFold by county) gain in
  R-squared from adding light on top of a model that already knows county size.
  Expected tiny for both productivity and prosperity.
- `productivity_ratio_flagged.csv` — the shared-denominator ratio specs, kept
  separate so no one mistakes them for the clean result.
- A figure, `productivity_vs_size.png` (or `prosperity_vs_size.png`): two
  scatters side by side, light vs total GDP (tight) and light vs GDP per worker
  (flat). That contrast is the whole argument in one picture.

## How to read the result

The expected, literature-consistent finding is that the strong size
relationship largely disappears for both productivity and prosperity, within
counties and out of sample. The verdict line the module prints says it plainly:
**light measures scale, not how productive or prosperous the people under it
are.** That is the sharpest possible confirmation that light is a size proxy,
and it is the honest, referee-proof version, not the naive ratio.

## A note on the data

Productivity here is GDP per **job** (BEA counts jobs, full-time plus
part-time, not full-time-equivalents), which is the standard county labor-input
measure. Prosperity is GDP per resident. Both are real; neither is synthetic.

The econometrics were verified on a synthetic panel built so that light tracks
size strongly and productivity only faintly (`test_productivity.py`); the module
correctly recovered a 0.95 size correlation, a near-zero productivity
correlation, and a tiny out-of-sample gain. The first run on your real BEA data
is the one that produces the numbers you will quote.
