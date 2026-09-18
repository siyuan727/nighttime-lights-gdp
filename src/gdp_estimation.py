"""
Reconstruct US GDP levels from nighttime lights and compare to the actual
figures, with explicit attention to the COVID-19 recession.

Why this is done at the county level and then aggregated:
national GDP for 2018 to 2020 is three numbers. You cannot fit a model to
three points. The county panel has thousands of units per year, so the model
is estimated there and the county predictions are summed to a national total.
That is the only version of "estimate US GDP" that has a real sample behind
it.

Design choices that hold bias down:
  - The model uses county fixed effects, so cross-county confounders (a place
    is bright and rich for reasons unrelated to the light-output link) are
    absorbed. The slope is identified from within-county variation over time,
    which is the defensible elasticity, not the inflated cross-sectional one.
  - No time effects are used, because time effects cannot be extrapolated to
    held-out years. Light and population carry the time variation instead.
  - Predictions are made in logs and converted back with Duan's smearing
    estimator, which corrects the low bias you get from a naive exp().
  - The reconstruction model is trained on pre-COVID years only, so 2020 is a
    genuine out-of-sample test. The gap between the light-based estimate and
    actual 2020 GDP is the part of the contraction that lights did not see.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging

from src.config import (
    OUTPUT_TAB, GDP_EST_TRAIN_END, GDP_EST_YEARS, COVID_YEAR, YEARS
)

log = logging.getLogger(__name__)

FEATURES = ["ln_ntl", "ln_pop"]


def _balanced_county_set(panel, needed_years):
    """Counties observed in every year we need, so national totals compare
    like with like across years."""
    have = panel.groupby("fips")["year"].agg(lambda s: set(needed_years).issubset(set(s)))
    return set(have[have].index)


def _within_fit(train, ycol, xcols):
    """
    County fixed-effects slope via the within (entity-demeaned) estimator.

    Returns the slope vector, clustered standard errors, the per-county
    intercepts, and Duan's smearing factor from the training residuals.
    """
    df = train.copy()
    grp = df.groupby("fips")

    # Demean outcome and regressors by county.
    y_dm = df[ycol] - grp[ycol].transform("mean")
    X_dm = pd.DataFrame(index=df.index)
    for c in xcols:
        X_dm[c] = df[c] - grp[c].transform("mean")

    # Slope from demeaned data (no intercept). Clustered SEs by county.
    res = sm.OLS(y_dm.values, X_dm.values).fit(
        cov_type="cluster", cov_kwds={"groups": df["fips"].values}
    )
    beta = pd.Series(res.params, index=xcols)
    se = pd.Series(res.bse, index=xcols)

    # County intercept: mean of (y - Xb) within county over training years.
    fitted_slope = sum(beta[c] * df[c] for c in xcols)
    df = df.assign(_resid_level=df[ycol] - fitted_slope)
    alpha = df.groupby("fips")["_resid_level"].mean()

    # Training residuals and Duan smearing factor for the log-to-level step.
    resid = df["_resid_level"] - df["fips"].map(alpha)
    smearing = float(np.mean(np.exp(resid)))

    return beta, se, alpha, smearing


def inject_synthetic_covid(panel, shock, seed=2020):
    """
    Demonstration only. Apply a 2020 GDP contraction on a copy of the panel
    while leaving nighttime lights untouched, so the light-based estimate
    overshoots actual 2020 the way it does on real data. GDP-derived columns
    are recomputed; the NTL columns are not.
    """
    rng = np.random.default_rng(seed)
    df = panel.copy()
    mask = df["year"] == COVID_YEAR
    # County-specific drop centred on `shock` with cross-county spread.
    drop = np.clip(rng.normal(shock, shock / 2.0, mask.sum()), 0.0, 0.6)
    df.loc[mask, "gdp_thousands"] = df.loc[mask, "gdp_thousands"] * (1 - drop)

    # Recompute GDP-based columns; do not touch ntl_mean / ln_ntl / ntl_growth.
    df["ln_gdp"] = np.log(df["gdp_thousands"].clip(1))
    if "population" in df.columns:
        df["gdp_pc"] = (df["gdp_thousands"] * 1000) / df["population"].clip(1)
        df["ln_gdp_pc"] = np.log(df["gdp_pc"].clip(1))
    df = df.sort_values(["fips", "year"])
    df["gdp_growth"] = df.groupby("fips")["ln_gdp"].diff()
    return df.reset_index(drop=True)


def estimate_us_gdp(panel, save=True, synthetic_covid=False, covid_shock=0.07):
    if synthetic_covid:
        log.info(f"Synthetic COVID demo: injecting ~{covid_shock:.0%} 2020 GDP "
                 f"drop, lights left unchanged.")
        panel = inject_synthetic_covid(panel, covid_shock)

    train_years = [y for y in YEARS if y <= GDP_EST_TRAIN_END]
    target_years = list(GDP_EST_YEARS)
    needed = train_years + target_years

    counties = _balanced_county_set(panel, needed)
    if len(counties) < 100:
        log.warning("Too few balanced counties for GDP level estimation.")
        return {}

    df = panel[panel["fips"].isin(counties)].copy()
    train = df[df["year"].isin(train_years)]

    log.info(f"GDP level estimation: train {train_years[0]}-{train_years[-1]}, "
             f"predict {target_years}, on {len(counties)} balanced counties.")

    beta, se, alpha, smearing = _within_fit(train, "ln_gdp", FEATURES)
    log.info(f"  within slope ln_ntl = {beta['ln_ntl']:.4f} "
             f"(SE {se['ln_ntl']:.4f}); smearing = {smearing:.4f}")

    # Reconstruct county GDP levels for each target year, then aggregate.
    rows = []
    for yr in target_years:
        sub = df[df["year"] == yr].copy()
        ln_hat = sub["fips"].map(alpha) + sum(beta[c] * sub[c] for c in FEATURES)
        gdp_hat = np.exp(ln_hat) * smearing            # thousands of USD
        nat_hat = float(gdp_hat.sum())
        nat_act = float(sub["gdp_thousands"].sum())
        rows.append({
            "year": yr,
            "national_gdp_actual_bil": round(nat_act / 1e6, 2),
            "national_gdp_estimated_bil": round(nat_hat / 1e6, 2),
            "pct_error": round((nat_hat / nat_act - 1) * 100, 2),
            "is_covid_year": yr == COVID_YEAR,
        })
    recon = pd.DataFrame(rows)

    covid = _covid_shock(df, train_years, target_years)

    log.info("\nNational GDP reconstruction (light-based vs actual):\n"
             + recon.to_string(index=False))
    if covid is not None:
        log.info(f"\nCOVID dummy (2020 log-GDP deviation net of lights): "
                 f"{covid['delta']:.4f} => {covid['pct']:.1f}% "
                 f"(SE {covid['se']:.4f}, p {covid['pval']:.3g})")

    if save:
        recon.to_csv(OUTPUT_TAB / "us_gdp_estimation.csv", index=False)
        if covid is not None:
            pd.DataFrame([covid]).to_csv(
                OUTPUT_TAB / "us_gdp_covid_shock.csv", index=False)

    return {"reconstruction": recon, "beta": beta, "se": se,
            "smearing": smearing, "covid": covid, "n_counties": len(counties)}


def _covid_shock(df, train_years, target_years):
    """
    Estimate the 2020 shock that lights do not explain. County fixed effects
    plus a 2020 indicator, over pre-COVID years through 2020. A negative
    coefficient means lights and population imply more GDP than actually
    occurred, that is, the proxy misses part of the contraction.
    """
    years = sorted(set(train_years + target_years))
    sub = df[df["year"].isin(years)].copy()
    if COVID_YEAR not in years:
        return None

    sub["d_covid"] = (sub["year"] == COVID_YEAR).astype(float)
    grp = sub.groupby("fips")
    cols = FEATURES + ["d_covid"]

    y_dm = sub["ln_gdp"] - grp["ln_gdp"].transform("mean")
    X_dm = pd.DataFrame({c: sub[c] - grp[c].transform("mean") for c in cols},
                        index=sub.index)
    res = sm.OLS(y_dm.values, X_dm.values).fit(
        cov_type="cluster", cov_kwds={"groups": sub["fips"].values})

    i = cols.index("d_covid")
    delta = float(res.params[i])
    return {"delta": round(delta, 4),
            "pct": round((np.exp(delta) - 1) * 100, 2),
            "se": round(float(res.bse[i]), 4),
            "pval": round(float(res.pvalues[i]), 4)}


def plot_us_gdp_estimation(results, save_path=None):
    import matplotlib.pyplot as plt
    from src.config import OUTPUT_FIG

    recon = results["reconstruction"]
    years = recon["year"].astype(str).tolist()
    actual = recon["national_gdp_actual_bil"].values
    est = recon["national_gdp_estimated_bil"].values

    x = np.arange(len(years))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, actual, w, label="Actual GDP", color="#33517D")
    ax.bar(x + w / 2, est, w, label="Light-based estimate", color="#C1443C")

    for i, r in recon.reset_index(drop=True).iterrows():
        ax.annotate(f"{r['pct_error']:+.1f}%",
                    (x[i] + w / 2, est[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9,
                    color="#7A2E31")
        if r["is_covid_year"]:
            top = max(actual[i], est[i])
            # Overshoot (positive error) is the expected COVID signature:
            # lights held up while output fell, so the estimate is too high.
            msg = ("COVID year:\nlights overstate GDP\n(miss the contraction)"
                   if r["pct_error"] > 0 else
                   "COVID year")
            ax.annotate(msg, (x[i], top), textcoords="offset points",
                        xytext=(0, 30), ha="center", fontsize=9,
                        color="#7A2E31",
                        arrowprops=dict(arrowstyle="->", color="#7A2E31"))

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("National GDP (USD billions)")
    ax.set_title("US GDP reconstructed from nighttime lights vs actual\n"
                 "county fixed-effects model, trained pre-COVID, "
                 "aggregated to national")
    ax.legend()
    ax.margins(y=0.15)
    fig.tight_layout()

    out = save_path or (OUTPUT_FIG / "us_gdp_estimation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out
