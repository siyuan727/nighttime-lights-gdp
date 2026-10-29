"""
Publication-ready descriptive statistics for the county panel.

Every empirical paper opens with a summary-statistics table, and a referee will
look for it before anything else. This module writes three things:

  1. descriptive_stats.csv    - N, mean, SD, min, p25, median, p75, max for the
                                variables used in the regressions, overall.
  2. descriptive_stats_by_group.csv - the same, split into urban and rural
                                counties, so the reader sees the two subsamples
                                the heterogeneity analysis relies on.
  3. correlation_matrix.csv   - pairwise correlations among the key variables,
                                which already previews the levels-versus-changes
                                contrast (ln_gdp vs ln_ntl high, growth vs
                                growth low).

It also logs the panel structure (counties, years, balanced or not), which a
referee wants stated explicitly.

Nothing here estimates anything; it only summarizes the panel built in
panel_construction.py, so it is safe to run any time after the panel exists.
"""

import logging

import numpy as np
import pandas as pd

from src.config import OUTPUT_TAB

log = logging.getLogger(__name__)

# Variables to summarize, in the order a reader expects them, with readable
# labels. Levels first, then logs, then growth rates.
LEVEL_VARS = [
    ("gdp_thousands", "Real GDP (thousands, chained $)"),
    ("ntl_mean",      "Mean radiance (nW/cm2/sr)"),
    ("population",    "Population"),
]
LOG_VARS = [
    ("ln_gdp", "log real GDP"),
    ("ln_ntl", "log radiance"),
    ("ln_pop", "log population"),
]
GROWTH_VARS = [
    ("gdp_growth", "GDP growth (Δ log)"),
    ("ntl_growth", "Radiance growth (Δ log)"),
]
ALL_VARS = LEVEL_VARS + LOG_VARS + GROWTH_VARS
CORR_VARS = ["ln_gdp", "ln_ntl", "ln_pop", "gdp_growth", "ntl_growth"]


def _summary(frame, varlist):
    """One summary row per variable: N, mean, SD, min, p25, p50, p75, max."""
    rows = []
    for col, label in varlist:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "variable": label,
            "N":      int(s.shape[0]),
            "mean":   s.mean(),
            "SD":     s.std(ddof=1),
            "min":    s.min(),
            "p25":    s.quantile(0.25),
            "median": s.median(),
            "p75":    s.quantile(0.75),
            "max":    s.max(),
        })
    out = pd.DataFrame(rows).set_index("variable")
    # Three decimal places for logs and growth; thousands separators are left to
    # the reader. Round to 3 to match the IPE table rule.
    return out.round(3)


def panel_structure(panel):
    """Facts a referee asks for: counties, years, observations, balance."""
    n_obs      = len(panel)
    n_counties = panel["fips"].nunique()
    years      = sorted(panel["year"].unique())
    per_county = panel.groupby("fips")["year"].nunique()
    balanced   = bool((per_county == len(years)).all())
    info = {
        "observations": n_obs,
        "counties": n_counties,
        "years": f"{years[0]}-{years[-1]}",
        "n_years": len(years),
        "balanced": balanced,
        "urban_share": round(float((panel["urban"] == 1).mean()), 3),
    }
    log.info("Panel structure: "
             f"{n_obs} obs, {n_counties} counties, {len(years)} years "
             f"({years[0]}-{years[-1]}), "
             f"{'balanced' if balanced else 'unbalanced'}.")
    return info


def build_descriptives(panel, save=True):
    """Write the summary-statistics and correlation tables; return them."""
    overall = _summary(panel, ALL_VARS)
    log.info("\nDescriptive statistics (overall):\n" + overall.to_string())

    # By urban / rural.
    groups = {}
    for label, val in [("urban", 1), ("rural", 0)]:
        sub = panel[panel["urban"] == val]
        if len(sub):
            groups[label] = _summary(sub, ALL_VARS)
    by_group = (pd.concat(groups, names=["group"])
                if groups else pd.DataFrame())

    # Correlation matrix among the regression variables.
    cors = [c for c in CORR_VARS if c in panel.columns]
    corr = panel[cors].corr().round(3)
    log.info("\nCorrelation matrix:\n" + corr.to_string())

    structure = panel_structure(panel)

    if save:
        OUTPUT_TAB.mkdir(parents=True, exist_ok=True)
        overall.to_csv(OUTPUT_TAB / "descriptive_stats.csv")
        if not by_group.empty:
            by_group.to_csv(OUTPUT_TAB / "descriptive_stats_by_group.csv")
        corr.to_csv(OUTPUT_TAB / "correlation_matrix.csv")
        pd.Series(structure).to_csv(OUTPUT_TAB / "panel_structure.csv",
                                    header=["value"])
        log.info(f"[tables] wrote descriptive_stats.csv, "
                 f"descriptive_stats_by_group.csv, correlation_matrix.csv, "
                 f"panel_structure.csv to {OUTPUT_TAB}")

    return {"overall": overall, "by_group": by_group,
            "correlation": corr, "structure": structure}
