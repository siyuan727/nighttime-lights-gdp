"""
Estimate GDP for a country from nighttime lights, using only real data.

Pipeline:
  1. Pick a panel of data-rich countries (by World Bank income group) with
     complete GDP and lights over the study years.
  2. Estimate the light-to-GDP elasticity on that panel with two-way (country
     and year) fixed effects, so the slope is a within-country relationship.
  3. Apply that elasticity to the target country's lights, anchored to one
     year of its actual World Bank GDP, and compare the reconstructed path to
     the target's real GDP as validation.
  4. Draw a lights map and a GDP map.

Sources: World Bank WDI (GDP), NASA Black Marble VNP46A4 (lights), Natural
Earth (boundaries). See wb_data.py and ntl_blackmarble.py.

The elasticity is learned on richer countries, so applying it to a poorer one
carries external-validity bias. The validation plot shows how large that error
actually is for the chosen target rather than assuming it away.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import wb_data, ntl_blackmarble
from src.config import (
    OUTPUT_TAB, OUTPUT_FIG, COUNTRY_YEARS, COUNTRY_ANCHOR_YEAR,
    COUNTRY_ELASTICITY_LOW, COUNTRY_ELASTICITY_HIGH,
    TARGET_ISO3, MAP_YEAR, PANEL_MAX_COUNTRIES, PANEL_INCOME_GROUPS,
)

log = logging.getLogger(__name__)


# ── Build the real panel ─────────────────────────────────────

def build_panel():
    meta = wb_data.get_country_metadata()

    rich = meta[meta["income"].isin(PANEL_INCOME_GROUPS)] \
        if "income" in meta.columns else meta
    rich_iso = sorted(rich["iso3"].unique())[:PANEL_MAX_COUNTRIES]
    wanted = sorted(set(rich_iso) | {TARGET_ISO3})
    log.info(f"Panel: {len(rich_iso)} data-rich countries + target {TARGET_ISO3}")

    geo = wb_data.get_country_geometries()
    geo = geo[geo["iso3"].isin(wanted)].copy()

    gdp = wb_data.get_wdi_gdp(wanted, COUNTRY_YEARS)
    ntl = ntl_blackmarble.get_country_ntl(geo, COUNTRY_YEARS)

    panel = gdp.merge(ntl, on=["iso3", "year"], how="inner")
    panel = panel[(panel["gdp"] > 0) & (panel["ntl_sum"] > 0)].copy()
    panel["ln_gdp"] = np.log(panel["gdp"])
    panel["ln_ntl"] = np.log(panel["ntl_sum"])

    # Keep countries with a complete series so the panel is balanced.
    complete = panel.groupby("iso3")["year"].nunique()
    full = complete[complete == len(COUNTRY_YEARS)].index
    panel = panel[panel["iso3"].isin(full)].reset_index(drop=True)

    rich_panel = panel[panel["iso3"] != TARGET_ISO3].copy()
    target_panel = panel[panel["iso3"] == TARGET_ISO3].copy()
    log.info(f"Balanced panel: {panel['iso3'].nunique()} countries "
             f"({rich_panel['iso3'].nunique()} for estimation), "
             f"target rows: {len(target_panel)}")
    return panel, rich_panel, target_panel, geo


# ── Estimation ───────────────────────────────────────────────

def estimate_elasticity(rich_panel):
    """Two-way (country and year) fixed-effects elasticity, clustered by country."""
    d = rich_panel.copy()
    cd = pd.get_dummies(d["iso3"], prefix="c", drop_first=True, dtype=float)
    yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[["ln_ntl"]], cd, yd], axis=1))
    res = sm.OLS(d["ln_gdp"].values, X.values).fit(
        cov_type="cluster", cov_kwds={"groups": d["iso3"].values})
    beta, se = float(res.params[1]), float(res.bse[1])
    log.info(f"Elasticity (two-way FE): {beta:.4f} (SE {se:.4f}), "
             f"{d['iso3'].nunique()} countries, {int(res.nobs)} obs")
    return beta, se


def estimate_target(target_panel, anchor_year, beta, se, elo, ehi):
    """Reconstruct target GDP from lights, anchored to its actual GDP in one year."""
    t = target_panel.sort_values("year").reset_index(drop=True).copy()
    if anchor_year not in set(t["year"]):
        anchor_year = int(t["year"].iloc[0])
    anchor_gdp = float(t.loc[t["year"] == anchor_year, "gdp"].iloc[0])
    ln_ntl_anchor = float(t.loc[t["year"] == anchor_year, "ln_ntl"].iloc[0])
    d_ntl = t["ln_ntl"] - ln_ntl_anchor
    ln_anchor = np.log(anchor_gdp)

    def path(b):
        return np.exp(ln_anchor + b * d_ntl)

    t["gdp_est"] = path(beta)
    lo, hi = path(beta - 1.96 * se), path(beta + 1.96 * se)
    t["gdp_ci_lo"], t["gdp_ci_hi"] = np.minimum(lo, hi), np.maximum(lo, hi)
    s_lo, s_hi = path(elo), path(ehi)
    t["gdp_sens_lo"], t["gdp_sens_hi"] = np.minimum(s_lo, s_hi), np.maximum(s_lo, s_hi)
    t["gdp_actual"] = t["gdp"]
    t["abs_pct_error"] = (t["gdp_est"] / t["gdp_actual"] - 1).abs() * 100
    return t, anchor_year


def run(save=True):
    panel, rich_panel, target_panel, geo = build_panel()
    if target_panel.empty:
        raise RuntimeError(f"No data for target {TARGET_ISO3}.")

    beta, se = estimate_elasticity(rich_panel)
    est, anchor = estimate_target(
        target_panel, COUNTRY_ANCHOR_YEAR, beta, se,
        COUNTRY_ELASTICITY_LOW, COUNTRY_ELASTICITY_HIGH)

    mape = float(est["abs_pct_error"].mean())
    log.info(f"Target {TARGET_ISO3}: light-based estimate vs actual WDI GDP, "
             f"mean abs error {mape:.1f}% (anchor {anchor})")

    cols = ["iso3", "year", "gdp_actual", "gdp_est", "gdp_ci_lo", "gdp_ci_hi",
            "gdp_sens_lo", "gdp_sens_hi", "abs_pct_error"]
    table = est[cols].copy()
    log.info("\n" + table.round(3).to_string(index=False))

    if save:
        table.to_csv(OUTPUT_TAB / "country_gdp_estimate_real.csv", index=False)

    return {"panel": panel, "rich_panel": rich_panel, "target": est,
            "geo": geo, "beta": beta, "se": se, "anchor": anchor, "mape": mape}


# ── Plots ────────────────────────────────────────────────────

def plot_estimate(results, save_path=None):
    import matplotlib.pyplot as plt
    est = results["target"]
    yrs = est["year"].values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(yrs, est["gdp_sens_lo"], est["gdp_sens_hi"],
                    color="#C1443C", alpha=0.12, label="elasticity sensitivity range")
    ax.fill_between(yrs, est["gdp_ci_lo"], est["gdp_ci_hi"],
                    color="#C1443C", alpha=0.28, label="95% CI (elasticity SE)")
    ax.plot(yrs, est["gdp_est"], color="#C1443C", lw=2, marker="o",
            label="light-based estimate")
    ax.plot(yrs, est["gdp_actual"], color="#33517D", lw=2, ls="--", marker="s",
            label="actual GDP (World Bank)")
    ax.axvline(results["anchor"], color="#555555", ls=":", lw=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP (constant 2015 USD)")
    ax.set_title(f"{TARGET_ISO3}: GDP estimated from nighttime lights vs actual\n"
                 f"elasticity from data-rich countries, anchored to {results['anchor']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / "country_gdp_estimate_real.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out


def plot_lights_and_gdp(results, year=None, save_path=None):
    """
    Ghana side by side: the actual nighttime-lights image on the left, and
    GDP against total luminosity over time on the right.

    Left is the real VIIRS radiance raster for the target country. Right shows
    the country's real World Bank GDP and its total Black Marble luminosity,
    both indexed to 100 at the anchor year, so the co-movement is visible on a
    common scale. If the raster download is unavailable, the figure falls back
    to the time-series panel alone.
    """
    import matplotlib.pyplot as plt

    year = year or MAP_YEAR
    panel, geo = results["panel"], results["geo"]
    one = geo[geo["iso3"] == TARGET_ISO3]

    gha = panel[panel["iso3"] == TARGET_ISO3].sort_values("year").copy()
    anchor = results["anchor"]

    def indexed(col):
        base = gha.loc[gha["year"] == anchor, col].iloc[0]
        return gha[col] / base * 100.0

    gha["gdp_idx"] = indexed("gdp")
    gha["ntl_idx"] = indexed("ntl_sum")

    # Try to fetch the real radiance raster for the left panel.
    da = None
    if not one.empty:
        try:
            ras = ntl_blackmarble.get_country_raster(one, year)
            da = ras.to_array().squeeze() if hasattr(ras, "to_array") else ras.squeeze()
        except Exception as e:
            log.warning(f"Lights raster unavailable, showing time series only: {e}")

    if da is not None:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5))
        da.plot.imshow(ax=axL, cmap="inferno", robust=True,
                       add_colorbar=True,
                       cbar_kwargs={"label": "radiance (nW/cm2/sr)", "shrink": 0.6})
        if not one.empty:
            one.boundary.plot(ax=axL, color="#66CCFF", linewidth=0.7)
        axL.set_title(f"{TARGET_ISO3} nighttime lights (VIIRS VNP46A4), {year}",
                      fontsize=11)
        axL.axis("off")
    else:
        fig, axR = plt.subplots(figsize=(9, 6))

    axR.plot(gha["year"], gha["gdp_idx"], color="#33517D", lw=2, marker="s",
             label="GDP (World Bank)")
    axR.plot(gha["year"], gha["ntl_idx"], color="#C1443C", lw=2, marker="o",
             label="total nighttime luminosity (Black Marble)")
    axR.axvline(anchor, color="#555555", ls=":", lw=1)
    axR.set_xlabel("Year")
    axR.set_ylabel(f"Index (anchor {anchor} = 100)")
    axR.set_title(f"{TARGET_ISO3}: GDP and luminosity over time", fontsize=11)
    axR.legend(fontsize=8)

    fig.suptitle(f"{TARGET_ISO3}: nighttime lights and GDP "
                 f"(real data: World Bank + NASA Black Marble)", fontsize=13)
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / "ghana_lights_and_gdp.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out
