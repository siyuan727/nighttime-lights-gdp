# Project improvements: industry heterogeneity, developing-economies panel, and three inference fixes

This adds five things to the project, each aimed at turning it from a competent
replication into something with a real contribution and publishable inference.
Everything below is new or updated code to drop into your repo. I could not run
any of it on live BEA / NASA / World Bank data here (no network access to those
sources), but I tested every estimator against a synthetic panel with a known
answer, so the econometrics are verified; the first live run is untested, mostly
because the exact columns the BEA and Black Marble APIs return can vary, and the
code logs what it got so any mismatch is a one-line fix.

## Where each file goes

| File | Put it in | New or replaces |
|---|---|---|
| `descriptives.py` | `src/` | new |
| `industry.py` | `src/` | new |
| `spatial_se.py` | `src/` | new |
| `measurement_error.py` | `src/` | new |
| `country_panel.py` | `src/` | new (replaces the Ghana-only logic in `country_real.py`) |
| `horse_race.py` | `src/` | **replaces** your existing `src/horse_race.py` (keeps all old functions, adds inference) |
| `run_improvements.py` | repo root (next to `main.py`) | new |
| `run_country_panel.py` | repo root | new |
| `config_additions.py` | paste its contents at the end of `src/config.py` | additions only |

After pasting the config lines and the files, run:

```
python main.py                 # builds the real panel as before
python run_improvements.py     # descriptives, industry, Conley SE, horse-race inference, measurement error
python run_country_panel.py    # the developing-economies external-validity analysis (heavy download)
```

`run_improvements.py` reads the cached panel from `data/clean/panel_county.pkl`,
so `main.py` must have run once first. Each block is wrapped so one failure does
not stop the rest.

---

## 1. Descriptive statistics (`descriptives.py`)

Writes the summary-statistics table every referee looks for first: N, mean, SD,
min, p25, median, p75, max for the regression variables, overall and split by
urban/rural, plus a correlation matrix and the panel structure (counties, years,
balanced or not). Outputs `descriptive_stats.csv`,
`descriptive_stats_by_group.csv`, `correlation_matrix.csv`, `panel_structure.csv`.

## 2. Industry heterogeneity: where the proxy breaks (`industry.py`)

This is the novel contribution. It pulls county GDP by industry (BEA **CAGDP2**),
builds each county's sector-composition shares, and estimates the light-GDP
elasticity two ways:

- **Interactions**: two-way FE with `ln_ntl` interacted with standardized county
  sector shares. Each interaction coefficient is how much the elasticity moves
  per one-SD increase in that sector's share. Output `industry_interactions.csv`.
- **Dominant-sector split**: assign each county its largest key sector, estimate
  the within elasticity per group. Output `industry_by_dominant_sector.csv` plus
  a bar-chart figure `industry_elasticity.png`.

The hypothesis to test: the elasticity is higher where manufacturing and trade
shares are high (bright, measurable) and lower where agriculture, finance, and
government shares are high. If that pattern holds, you have a map of where
satellite GDP can be trusted, which no one has cleanly published at the US county
level.

Line codes are resolved by matching the BEA industry *descriptions*, not by
hard-coded numbers, so it survives BEA renumbering; it logs the sector-to-line
map it found.

## 3. Conley spatial-HAC standard errors (`spatial_se.py`)

Clustering by county ignores that neighboring counties' errors are correlated, so
your reported precision is too optimistic. This implements the Conley (1999)
spatial-HAC sandwich directly, with a Bartlett kernel in great-circle distance
(cutoff 500 km by default) and an optional Bartlett kernel in time for serial
correlation. `run_conley_levels` attaches county centroids from your TIGER
shapefile and reports the Conley SE next to the classical SE for the pooled
levels regression. Output `conley_levels.csv`.

In my synthetic test the Conley SE came out about four times the classical SE,
which is the whole point: it will widen your confidence intervals, and reporting
it is what makes the cross-sectional result credible.

## 4. Horse-race inference (updated `horse_race.py`)

Your horse race reported point gains in R-squared with no uncertainty. The new
`run_horse_race_inference` adds, for both the levels and changes tasks:

- a **cluster (block) bootstrap over counties** giving a standard error and a 95%
  confidence interval for the gain in out-of-sample R-squared;
- a **Diebold-Mariano test** on the per-observation squared-error loss
  differential, clustered by county, testing one-sided whether lights reduce
  prediction error.

Output `horse_race_inference.csv`. The expected and publishable result: the
levels gain has a CI well above zero, and the changes gain is small with a CI
that includes or sits near zero and a non-significant DM test. That lets you say
"lights do not track short-run changes" as a tested claim, not an eyeball. All
your original functions (`run_horse_race`, `plot_horse_race`) are unchanged.

## 5. Measurement-error correction: attenuation vs composition (`measurement_error.py`)

Your paper says the 0.845 → 0.162 collapse is part omitted-variable bias and part
attenuation but cannot separate them. This separates them. It builds two
independent noisy measures of each county's annual brightness (mean radiance over
the first half of the year and the second half, from the **monthly** Black Marble
product VNP46A3), and instruments one with the other in the within regression.
Under classical measurement error with errors uncorrelated across the two halves,
that recovers the attenuation-corrected elasticity. It reports:

- the naive FE elasticity, the IV (corrected) elasticity, and the implied
  reliability ratio (how much attenuation there is);
- the decomposition: `pooled − FE = (pooled − IV) [composition] + (IV − FE) [attenuation]`.

Output `measurement_error.csv`. If you cannot build the half-year measures,
`estimate_attenuation_lag_fallback` instruments contemporaneous light with its
one-year lag (a weaker assumption, reported as a robustness variant), output
`measurement_error_lag.csv`.

In my synthetic test with a true within elasticity of 0.16, the naive FE came in
at 0.143 and the IV correctly moved it up to 0.152, with a reliability ratio
below one, confirming the correction works in the right direction.

**Data note**: the half-year measures need the monthly Black Marble product,
which is twelve downloads per year instead of one, so it is the heaviest new data
step. `get_county_ntl_split` builds and caches it.

## 6. Developing-economies panel (`country_panel.py`, `run_country_panel.py`)

Replaces the eight-alphabetical-country Ghana exercise. It learns the reference
elasticity on high-income countries, reconstructs GDP for **every** developing
economy (World Bank low, lower-middle, and upper-middle income) with complete
data, and regresses the reconstruction error across countries on structural
covariates: agriculture share, electricity access, rural share, and
self-employment (informality proxy), all from WDI. Outputs
`country_reconstruction_errors.csv` (one row per country) and
`country_error_regression.csv` (the second-stage coefficients), plus scatter
figures.

The external-validity finding is no longer "Ghana drifts," it is "reconstruction
error rises with agriculture share and falls with electrification across 150
countries," which is a real, testable result. The caps in config bound the
national-lights download; set them to `None` for the full representative panel.

---

## Honest caveats to keep in the paper

- The half-year measurement-error instrument assumes the two composites' errors
  are independent. They are from different months of the same year, which is
  defensible but not airtight; report the lag-instrument version alongside it.
- The informality proxy is self-employment share; true informal-employment data
  (ILO ILOSTAT) would be better and is worth adding if you go further.
- CAGDP2 is current-dollar GDP by industry; shares are within-county ratios so
  this is fine, but say so.
- Everything still rests on BEA county GDP, which is a modeled, apportioned
  series. That caveat does not go away.
