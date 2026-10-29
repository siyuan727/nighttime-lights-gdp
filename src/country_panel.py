"""
External validity done properly: a representative panel of developing economies.

The old country module learned an elasticity on eight alphabetically chosen
countries and applied it to Ghana. That is not defensible: eight countries is
too few, and picking them by the alphabet is arbitrary. This module replaces it
with a real cross-country exercise and turns the external-validity question into
a testable one.

Design:
  1. Learn a reference light-GDP elasticity on the data-rich (high-income)
     countries with two-way (country and year) fixed effects.
  2. For every developing economy (World Bank low, lower-middle, and upper-middle
     income) with complete GDP and lights, do two things:
       - estimate the country's OWN within elasticity (time-series slope of
         log GDP on log lights), and
       - reconstruct its GDP from lights using the reference elasticity, anchored
         to one year of its real World Bank GDP, and record the mean absolute
         percentage error and the mean signed error (positive = lights overstate).
  3. SECOND STAGE: regress the reconstruction error across countries on
     structural covariates that should govern how well lights track output:
       - agriculture value added (% GDP)          dim, land-based output
       - access to electricity (% population)      lights require electrification
       - rural population (% total)                dispersed, low-radiance activity
       - self-employment (% employment)            informality proxy
     The hypothesis, stated so the data can reject it: reconstruction error is
     larger in more agricultural, less electrified, more rural, and more informal
     economies. That relationship, not Ghana alone, is the external-validity
     finding.

Sources: World Bank WDI (GDP and all covariates), NASA Black Marble (national
lights), Natural Earth (boundaries). Downloading national lights for ~150
countries is the heavy step; it caches, and the per-group caps in config bound
it if needed.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
    OUTPUT_TAB, COUNTRY_YEARS, COUNTRY_ANCHOR_YEAR,
    REFERENCE_INCOME_GROUPS, DEV_INCOME_GROUPS, COUNTRY_COVARIATES,
    REFERENCE_MAX_COUNTRIES, DEV_MAX_COUNTRIES,
)

# wb_data and ntl_blackmarble pull in geopandas, wbgapi and blackmarblepy, which
# are only needed for the data-fetching functions. They are imported inside those
# functions so the estimators (reference_elasticity, build_error_table,
# error_regressions) can be used and tested without the heavy geo stack.

log = logging.getLogger(__name__)


# ── Country selection and data ───────────────────────────────

def select_groups():
    """Reference (data-rich) and developing ISO3 lists from WB income groups."""
    from src import wb_data
    meta = wb_data.get_country_metadata()
    if "income" not in meta.columns:
        raise RuntimeError("World Bank metadata has no income column.")
    ref = sorted(meta.loc[meta["income"].isin(REFERENCE_INCOME_GROUPS), "iso3"].unique())
    dev = sorted(meta.loc[meta["income"].isin(DEV_INCOME_GROUPS), "iso3"].unique())
    if REFERENCE_MAX_COUNTRIES:
        ref = ref[:REFERENCE_MAX_COUNTRIES]
    if DEV_MAX_COUNTRIES:
        dev = dev[:DEV_MAX_COUNTRIES]
    log.info(f"Reference (data-rich) countries: {len(ref)}; "
             f"developing countries: {len(dev)}")
    return ref, dev, meta


def get_wdi_covariates(iso3_list, years=None):
    """
    Structural covariates per country, averaged over the sample years, from WDI.
    Returns one row per iso3 with the columns named in COUNTRY_COVARIATES.
    """
    import wbgapi as wb
    years = list(years or COUNTRY_YEARS)
    frames = []
    for name, code in COUNTRY_COVARIATES.items():
        try:
            raw = wb.data.DataFrame(code, economy=iso3_list, time=years,
                                    skipAggs=True, labels=False).reset_index()
        except Exception as e:
            log.warning(f"WDI covariate {name} ({code}) failed: {e}")
            continue
        id_col = raw.columns[0]
        long = raw.melt(id_vars=id_col, var_name="period", value_name=name)
        long = long.rename(columns={id_col: "iso3"})
        long["iso3"] = long["iso3"].astype(str).str.upper()
        # Country mean over available years.
        m = long.groupby("iso3")[name].mean().reset_index()
        frames.append(m)
    if not frames:
        raise RuntimeError("No WDI covariates could be fetched.")
    cov = frames[0]
    for f in frames[1:]:
        cov = cov.merge(f, on="iso3", how="outer")
    log.info(f"WDI covariates for {cov['iso3'].nunique()} countries: "
             + ", ".join([c for c in cov.columns if c != 'iso3']))
    return cov


def build_country_panel(iso3_list, years=None, label=""):
    """Balanced [iso3, year, gdp, ntl_sum, ln_gdp, ln_ntl] for the given ISO3s."""
    from src import wb_data, ntl_blackmarble
    years = list(years or COUNTRY_YEARS)
    geo = wb_data.get_country_geometries()
    geo = geo[geo["iso3"].isin(iso3_list)].copy()
    gdp = wb_data.get_wdi_gdp(iso3_list, years)
    ntl = ntl_blackmarble.get_country_ntl(geo, years)

    panel = gdp.merge(ntl, on=["iso3", "year"], how="inner")
    panel = panel[(panel["gdp"] > 0) & (panel["ntl_sum"] > 0)].copy()
    panel["ln_gdp"] = np.log(panel["gdp"])
    panel["ln_ntl"] = np.log(panel["ntl_sum"])

    complete = panel.groupby("iso3")["year"].nunique()
    full = complete[complete == len(years)].index
    panel = panel[panel["iso3"].isin(full)].reset_index(drop=True)
    log.info(f"{label} panel: {panel['iso3'].nunique()} countries with complete "
             f"series, {len(panel)} rows.")
    return panel


# ── Estimation ───────────────────────────────────────────────

def reference_elasticity(reference_panel):
    """Two-way FE light-GDP elasticity on the reference panel, clustered by country."""
    d = reference_panel.copy()
    cd = pd.get_dummies(d["iso3"], prefix="c", drop_first=True, dtype=float)
    yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[["ln_ntl"]], cd, yd], axis=1))
    res = sm.OLS(d["ln_gdp"].values, X.values).fit(
        cov_type="cluster", cov_kwds={"groups": d["iso3"].values})
    beta, se = float(res.params[1]), float(res.bse[1])
    log.info(f"Reference elasticity (two-way FE): {beta:.4f} (SE {se:.4f}), "
             f"{d['iso3'].nunique()} countries, {int(res.nobs)} obs")
    return beta, se


def own_elasticity(country_df):
    """
    A single country's within elasticity: OLS slope of ln_gdp on ln_ntl after
    removing the country's own means (a demeaned time-series slope). Returns NaN
    if lights do not vary.
    """
    x = country_df["ln_ntl"].values
    y = country_df["ln_gdp"].values
    x = x - x.mean(); y = y - y.mean()
    vx = float(x @ x)
    return float((x @ y) / vx) if vx > 0 else np.nan


def reconstruct_country(country_df, beta, anchor_year):
    """
    Reconstruct GDP from lights using the reference beta, anchored to one actual
    GDP year. Returns (mape, mean_signed_error_pct). Positive signed error means
    lights overstate GDP, the expected direction when lights outrun output.
    """
    t = country_df.sort_values("year").reset_index(drop=True)
    yrs = set(t["year"])
    anchor = anchor_year if anchor_year in yrs else int(t["year"].iloc[0])
    ln_ntl_anchor = float(t.loc[t["year"] == anchor, "ln_ntl"].iloc[0])
    anchor_gdp = float(t.loc[t["year"] == anchor, "gdp"].iloc[0])
    est = np.exp(np.log(anchor_gdp) + beta * (t["ln_ntl"] - ln_ntl_anchor))
    signed = (est / t["gdp"] - 1.0) * 100.0
    return float(signed.abs().mean()), float(signed.mean())


def build_error_table(developing_panel, beta, anchor_year, covariates):
    """One row per developing country: own elasticity, reconstruction error, covariates."""
    rows = []
    for iso, sub in developing_panel.groupby("iso3"):
        mape, signed = reconstruct_country(sub, beta, anchor_year)
        rows.append({"iso3": iso,
                     "own_elasticity": round(own_elasticity(sub), 4),
                     "recon_mape": round(mape, 3),
                     "recon_signed_error": round(signed, 3),
                     "years": int(sub["year"].nunique())})
    err = pd.DataFrame(rows).merge(covariates, on="iso3", how="left")
    log.info(f"Error table: {len(err)} developing countries.")
    return err


def error_regressions(error_table, save=True):
    """
    Second stage: regress reconstruction error (and signed error, and the gap
    between own and reference elasticity) on the structural covariates, with
    heteroskedasticity-robust SE. This is the external-validity result.
    """
    covs = [c for c in COUNTRY_COVARIATES.keys() if c in error_table.columns]
    results = {}
    out_rows = []
    for dep in ["recon_mape", "recon_signed_error"]:
        d = error_table.dropna(subset=[dep] + covs).copy()
        if len(d) < len(covs) + 3:
            log.warning(f"Too few countries ({len(d)}) to regress {dep}.")
            continue
        X = sm.add_constant(d[covs].astype(float))
        res = sm.OLS(d[dep].astype(float).values, X.values).fit(cov_type="HC1")
        results[dep] = res
        names = ["const"] + covs
        for nm, b, se, p in zip(names, res.params, res.bse, res.pvalues):
            out_rows.append({"dependent": dep, "term": nm,
                             "coef": round(float(b), 4),
                             "se": round(float(se), 4),
                             "pval": round(float(p), 4)})
        log.info(f"{dep} ~ covariates: n={int(res.nobs)}, R2={res.rsquared:.3f}")

    table = pd.DataFrame(out_rows)
    log.info("\nReconstruction error on structural covariates:\n"
             + table.to_string(index=False))
    if save and not table.empty:
        table.to_csv(OUTPUT_TAB / "country_error_regression.csv", index=False)
    return {"models": results, "table": table}


def run(save=True):
    """Full developing-economies external-validity pipeline."""
    ref_iso, dev_iso, _meta = select_groups()

    ref_panel = build_country_panel(ref_iso, label="reference")
    dev_panel = build_country_panel(dev_iso, label="developing")
    if ref_panel.empty or dev_panel.empty:
        raise RuntimeError("Reference or developing panel is empty.")

    beta, se = reference_elasticity(ref_panel)

    covariates = get_wdi_covariates(sorted(dev_panel["iso3"].unique()))
    err = build_error_table(dev_panel, beta, COUNTRY_ANCHOR_YEAR, covariates)
    reg = error_regressions(err, save=save)

    if save:
        err.to_csv(OUTPUT_TAB / "country_reconstruction_errors.csv", index=False)
        log.info(f"[tables] wrote country_reconstruction_errors.csv and "
                 f"country_error_regression.csv to {OUTPUT_TAB}")

    summary = {
        "reference_beta": beta, "reference_se": se,
        "n_reference": int(ref_panel["iso3"].nunique()),
        "n_developing": int(dev_panel["iso3"].nunique()),
        "mean_mape": float(err["recon_mape"].mean()),
        "mean_signed_error": float(err["recon_signed_error"].mean()),
    }
    log.info(f"Developing-economy reconstruction: mean MAPE "
             f"{summary['mean_mape']:.1f}%, mean signed error "
             f"{summary['mean_signed_error']:+.1f}% across "
             f"{summary['n_developing']} countries.")
    return {"reference_panel": ref_panel, "developing_panel": dev_panel,
            "beta": beta, "errors": err, "regression": reg, "summary": summary}


def plot_error_vs_covariate(error_table, covariate="agriculture_share",
                            save_path=None):
    """Scatter of reconstruction error against one structural covariate."""
    import matplotlib.pyplot as plt
    from src.config import OUTPUT_FIG, NAVY, MAROON

    d = error_table.dropna(subset=["recon_mape", covariate])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(d[covariate], d["recon_mape"], color=NAVY, alpha=0.7)
    if len(d) >= 3:
        b1, b0 = np.polyfit(d[covariate], d["recon_mape"], 1)
        xs = np.linspace(d[covariate].min(), d[covariate].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=MAROON, lw=2)
    ax.set_xlabel(covariate.replace("_", " "))
    ax.set_ylabel("Reconstruction error (mean abs %)")
    ax.set_title("Where borrowed light-GDP elasticities fail\n"
                 "Developing-economy reconstruction error vs structure")
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / f"country_error_vs_{covariate}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out
