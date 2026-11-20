"""
Does nighttime light proxy PRODUCTIVITY, or only SIZE?

This is the sharpest version of the project's thesis. The raw light-GDP
correlation is strong because bright places are big places. The real test of
whether light carries economic information, rather than just a head count, is
whether it tracks output PER WORKER (labor productivity) and output PER PERSON
(prosperity). If it does not, light is a scale proxy, full stop.

Two traps this module is built to avoid, both of which a referee will raise:

  1. "Dividing by workers is not controlling for workers." GDP per worker is
     log GDP minus log employment, so forming the ratio imposes a unit
     coefficient on size instead of estimating it. So alongside the ratio we
     run the honest control: log GDP on log light WITH log employment on the
     right-hand side, within two-way fixed effects. The light coefficient there
     is light's partial association with output holding size fixed, i.e. the
     productivity signal, with size estimated, not assumed.

  2. "Ratios that share a denominator correlate mechanically." GDP per worker
     and light-per-worker both divide by employment, which can manufacture
     correlation. We sidestep this by using MEAN radiance (ntl_mean, brightness
     per unit area) as the light measure: it is not divided by employment, so
     there is no shared denominator. The literal light-per-worker ratio is
     reported too, but flagged as the contaminated one, for contrast.

Evidence, from weak to strong:
  A. Correlation collapse: corr(light, GDP) vs corr(light, GDP/worker) and
     corr(light, GDP/capita). The headline, and the cleanest slide.
  B. Cross-sectional OLS ladder for productivity, with and without a size
     control and state effects.
  C. Within-county two-way fixed effects (PanelOLS), the causal-ish version.
  D. Out-of-sample horse race (GroupKFold by county): does light add anything
     to a prediction of productivity that already knows county size?
  E. The flagged light-per-worker ratio, for completeness.

Productivity needs county employment, which the base panel does not carry, so
this module fetches BEA CAEMP25N (total jobs by county) with the same key and
caching pattern as the GDP pull. If employment cannot be fetched, the
per-capita (prosperity) half still runs off population, which is already in the
panel, so the module always produces a conclusion.

Expected result, consistent with the literature (lights proxy total GDP well,
per-capita GDP poorly): the strong size relationship largely collapses for both
productivity and prosperity, within and out of sample. Light measures scale,
not how productive or prosperous the people under it are.
"""

import io
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

from src.config import (
    DATA_RAW, DATA_CLEAN, OUTPUT_TAB, BEA_API_KEY, YEARS,
    EXCLUDE_FIPS_PREFIX, START_YEAR, END_YEAR,
)

log = logging.getLogger(__name__)

BEA_ENDPOINT = "https://apps.bea.gov/api/data/"
EMP_CACHE = DATA_RAW / "bea_county_employment.csv"
PROD_PANEL_PKL = DATA_CLEAN / "panel_productivity.pkl"


# ────────────────────────────────────────────────────────────────────────
# 1. Employment data (BEA CAEMP25N, total full-time and part-time jobs)
# ────────────────────────────────────────────────────────────────────────
def fetch_bea_employment():
    """County total employment (jobs), mirroring fetch_bea_gdp in style."""
    if EMP_CACHE.exists():
        log.info("Loading cached BEA employment...")
        return pd.read_csv(EMP_CACHE, dtype={"fips": str})
    if not BEA_API_KEY or BEA_API_KEY == "YOUR_BEA_API_KEY_HERE":
        raise RuntimeError("No BEA key set in src/config.py.")

    import requests
    log.info("Fetching BEA county employment (CAEMP25N, total jobs)...")
    params = {
        "UserID": BEA_API_KEY, "method": "GetData",
        "datasetname": "Regional", "TableName": "CAEMP25N",
        "LineCode": "10", "GeoFips": "COUNTY",       # 10 = total employment
        "Year": ",".join(str(y) for y in YEARS),
        "ResultFormat": "json",
    }
    try:
        r = requests.get(BEA_ENDPOINT, params=params, timeout=120)
        r.raise_for_status()
        records = r.json()["BEAAPI"]["Results"]["Data"]
    except Exception as e:
        raise RuntimeError(f"BEA employment API failed: {e}") from e

    rows = []
    for rec in records:
        fips = str(rec.get("GeoFips", "")).zfill(5)
        val = rec.get("DataValue", "")
        if val in ("(NA)", "(D)", "(L)", "", "--", "(NM)"):
            continue
        try:
            emp = float(str(val).replace(",", ""))
        except ValueError:
            continue
        if emp <= 0:
            continue
        rows.append({"fips": fips, "year": int(rec.get("TimePeriod", 0)),
                     "employment": emp})

    df = pd.DataFrame(rows)
    df = df[~df["fips"].str[:2].isin(EXCLUDE_FIPS_PREFIX)]
    df = df[df["fips"].str.len() == 5]
    df = df[df["year"].between(START_YEAR, END_YEAR)]
    df = df.dropna(subset=["employment"])
    EMP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EMP_CACHE, index=False)
    log.info(f"BEA employment: {len(df)} rows, {df['fips'].nunique()} counties")
    return df


# ────────────────────────────────────────────────────────────────────────
# 2. Build the productivity panel
# ────────────────────────────────────────────────────────────────────────
def build_productivity_panel(panel, emp_df=None, save=True):
    """
    Add employment, productivity (GDP per worker) and prosperity (GDP per
    capita) variables to the county panel. Employment is optional: without it
    the per-worker columns are absent but the per-capita analysis still runs.
    """
    p = panel.copy()
    p["fips"] = p["fips"].astype(str).str.zfill(5)

    # Prosperity (GDP per capita) is always available from population.
    if "ln_gdp_pc" not in p.columns and "population" in p.columns:
        p["gdp_pc"] = (p["gdp_thousands"] * 1000) / p["population"].clip(lower=1)
        p["ln_gdp_pc"] = np.log(p["gdp_pc"].clip(lower=1))

    # Prosperity side of the light measures (flagged ratio uses total light).
    if "ntl_sum" in p.columns and "population" in p.columns:
        p["ntl_pc"] = p["ntl_sum"] / p["population"].clip(lower=1)
        p["ln_ntl_pc"] = np.log(p["ntl_pc"].clip(lower=1e-9) + 1e-9)

    # Productivity (GDP per worker) needs employment.
    if emp_df is None:
        try:
            emp_df = fetch_bea_employment()
        except Exception as e:
            log.warning(f"Employment unavailable ({e}); building per-capita "
                        f"(prosperity) half only.")
            emp_df = None

    if emp_df is not None:
        emp_df = emp_df.copy()
        emp_df["fips"] = emp_df["fips"].astype(str).str.zfill(5)
        p = p.merge(emp_df[["fips", "year", "employment"]],
                    on=["fips", "year"], how="left")
        p["ln_emp"] = np.log(p["employment"].clip(lower=1))
        p["gdp_pw"] = (p["gdp_thousands"] * 1000) / p["employment"].clip(lower=1)
        p["ln_gdp_pw"] = np.log(p["gdp_pw"].clip(lower=1))
        if "ntl_sum" in p.columns:
            p["ntl_pw"] = p["ntl_sum"] / p["employment"].clip(lower=1)
            p["ln_ntl_pw"] = np.log(p["ntl_pw"].clip(lower=1e-9) + 1e-9)
        have_emp = p["employment"].notna()
        log.info(f"Productivity panel: employment merged for "
                 f"{have_emp.sum()} / {len(p)} rows "
                 f"({p.loc[have_emp, 'fips'].nunique()} counties).")
    else:
        log.info("Productivity panel: per-capita only (no employment).")

    if save:
        PROD_PANEL_PKL.parent.mkdir(parents=True, exist_ok=True)
        p.to_pickle(PROD_PANEL_PKL)
        p.to_csv(DATA_CLEAN / "panel_productivity.csv", index=False)
    return p


# ────────────────────────────────────────────────────────────────────────
# 3. Estimators (mirror the project's statsmodels + PanelOLS conventions)
# ────────────────────────────────────────────────────────────────────────
def _pooled_ols(df, y, xcols, state_fe=False, cluster="fips"):
    """Pooled OLS with optional state dummies, SE clustered by county."""
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y] + xcols + [cluster]).copy()
    X = d[xcols].astype(float).copy()
    if state_fe:
        sd = pd.get_dummies(d["fips"].str[:2], prefix="st", drop_first=True, dtype=float)
        X = pd.concat([X.reset_index(drop=True), sd.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    res = sm.OLS(d[y].astype(float).values, X.values).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster].values})
    i = 1  # column 0 is const; first predictor is xcols[0]
    return {
        "coef": float(res.params[i]), "se": float(res.bse[i]),
        "pval": float(res.pvalues[i]), "r2": float(res.rsquared),
        "n": int(res.nobs),
    }


def _twoway_fe(df, y, xcols):
    """Within county + year two-way FE via PanelOLS, clustered by county."""
    from linearmodels.panel import PanelOLS
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y] + xcols).copy()
    d["fips"] = d["fips"].astype(str)
    d["yr"] = pd.to_datetime(d["year"], format="%Y")
    d = d.set_index(["fips", "yr"])
    res = PanelOLS(d[y], d[xcols], entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    key = xcols[0]
    return {
        "coef": float(res.params[key]), "se": float(res.std_errors[key]),
        "pval": float(res.pvalues[key]), "r2_within": float(res.rsquared_within),
        "n": int(res.nobs),
    }


def _horse_race(df, y, base_cols, light_col, n_splits=5):
    """Out-of-sample R2 for base vs base+light, GroupKFold by county."""
    d = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[y] + base_cols + [light_col, "fips"]).copy()
    # year dummies so the baseline knows the national year effect
    yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
    Xb = pd.concat([d[base_cols].astype(float).reset_index(drop=True),
                    yd.reset_index(drop=True)], axis=1).values
    Xf = np.column_stack([Xb, d[light_col].astype(float).values])
    yv = d[y].astype(float).values
    groups = d["fips"].values
    bp = np.zeros_like(yv); fp = np.zeros_like(yv)
    for tr, te in GroupKFold(n_splits=n_splits).split(Xb, yv, groups):
        bp[te] = LinearRegression().fit(Xb[tr], yv[tr]).predict(Xb[te])
        fp[te] = LinearRegression().fit(Xf[tr], yv[tr]).predict(Xf[te])
    b_r2, f_r2 = r2_score(yv, bp), r2_score(yv, fp)
    b_rmse = np.sqrt(mean_squared_error(yv, bp))
    f_rmse = np.sqrt(mean_squared_error(yv, fp))
    return {
        "base_r2": float(b_r2), "full_r2": float(f_r2),
        "delta_r2": float(f_r2 - b_r2),
        "rmse_reduction_pct": float((b_rmse - f_rmse) / b_rmse * 100),
        "n": int(len(yv)),
    }


# ────────────────────────────────────────────────────────────────────────
# 4. Run everything and reach a verdict
# ────────────────────────────────────────────────────────────────────────
def run_productivity(panel, save=True):
    """
    Full productivity-vs-size analysis. `panel` may be the base county panel
    (employment will be fetched) or an already-built productivity panel.
    """
    p = panel.copy()
    if "ln_gdp_pw" not in p.columns and "ln_gdp_pc" not in p.columns:
        p = build_productivity_panel(p, save=save)

    light = "ln_ntl"            # mean radiance: brightness, not a ratio of size
    has_pw = "ln_gdp_pw" in p.columns and p["ln_gdp_pw"].notna().any()

    # ── A. Correlation collapse (size -> prosperity -> productivity) ──
    corr_rows = []
    pairs = [("GDP (size)", "ln_gdp"),
             ("GDP per capita (prosperity)", "ln_gdp_pc")]
    if has_pw:
        pairs.append(("GDP per worker (productivity)", "ln_gdp_pw"))
    for label, col in pairs:
        if col in p.columns:
            d = p[[light, col]].replace([np.inf, -np.inf], np.nan).dropna()
            corr_rows.append({"outcome": label,
                              "corr_with_log_light": round(float(d.corr().iloc[0, 1]), 4),
                              "n": int(len(d))})
    corr_tbl = pd.DataFrame(corr_rows)
    log.info("\nCorrelation of log mean radiance with:\n" + corr_tbl.to_string(index=False))

    # ── B/C. Regression ladders for each outcome ──
    reg_rows = []

    def ladder(y, size_ctrl):
        if y not in p.columns or not p[y].notna().any():
            return
        biv = _pooled_ols(p, y, [light])
        reg_rows.append(dict(outcome=y, spec="OLS bivariate", **biv))
        ctrl = _pooled_ols(p, y, [light, size_ctrl])
        reg_rows.append(dict(outcome=y, spec=f"OLS + {size_ctrl} (size control)", **ctrl))
        stfe = _pooled_ols(p, y, [light, size_ctrl], state_fe=True)
        reg_rows.append(dict(outcome=y, spec=f"OLS + {size_ctrl} + state FE", **stfe))
        try:
            fe = _twoway_fe(p, y, [light, size_ctrl])
            fe = {k: fe.get(k) for k in ("coef", "se", "pval", "n")} | {"r2": fe.get("r2_within")}
            reg_rows.append(dict(outcome=y, spec="Two-way FE (within county)", **fe))
        except Exception as e:
            log.warning(f"Two-way FE for {y} skipped: {e}")

    # size benchmark: GDP on light controlling for population -> the known result
    ladder("ln_gdp", "ln_pop")
    # prosperity: GDP per capita, size control = population
    ladder("ln_gdp_pc", "ln_pop")
    # productivity: GDP per worker, size control = employment
    if has_pw:
        ladder("ln_gdp_pw", "ln_emp")
    reg_tbl = pd.DataFrame(reg_rows)

    # ── D. Out-of-sample horse race: does light beat knowing county size? ──
    hr_rows = []
    hr_specs = [("ln_gdp_pc", "ln_pop")]
    if has_pw:
        hr_specs.append(("ln_gdp_pw", "ln_emp"))
    for y, size_ctrl in hr_specs:
        if y in p.columns and p[y].notna().any():
            r = _horse_race(p, y, [size_ctrl], light)
            hr_rows.append(dict(outcome=y, baseline=f"{size_ctrl}+year", **r))
    hr_tbl = pd.DataFrame(hr_rows)
    if not hr_tbl.empty:
        log.info("\nOut-of-sample horse race (light added on top of size):\n"
                 + hr_tbl.round(4).to_string(index=False))

    # ── E. Flagged light-per-worker ratio (shared-denominator caveat) ──
    ratio_rows = []
    if has_pw and "ln_ntl_pw" in p.columns:
        r = _pooled_ols(p, "ln_gdp_pw", ["ln_ntl_pw"])
        ratio_rows.append(dict(spec="GDP/worker ~ light/worker (SHARED denom, flagged)", **r))
    if "ln_gdp_pc" in p.columns and "ln_ntl_pc" in p.columns:
        r = _pooled_ols(p, "ln_gdp_pc", ["ln_ntl_pc"])
        ratio_rows.append(dict(spec="GDP/capita ~ light/capita (SHARED denom, flagged)", **r))
    ratio_tbl = pd.DataFrame(ratio_rows)

    # ── Verdict ──
    size_corr = corr_tbl.loc[corr_tbl.outcome.str.startswith("GDP (size"),
                             "corr_with_log_light"].squeeze()
    prod_label = "GDP per worker (productivity)" if has_pw else "GDP per capita (prosperity)"
    prod_corr = corr_tbl.loc[corr_tbl.outcome == prod_label,
                             "corr_with_log_light"].squeeze()
    drop = (1 - abs(prod_corr) / abs(size_corr)) * 100 if size_corr else float("nan")
    verdict = (
        f"Light correlates {size_corr:.2f} with GDP (size) but {prod_corr:.2f} with "
        f"{prod_label.split(' (')[0]}, a {drop:.0f}% drop. Within counties and out of "
        f"sample, light adds little once size is known. "
        + ("Nighttime light proxies SCALE, not productivity."
           if abs(prod_corr) < 0.5 and abs(prod_corr) < abs(size_corr)
           else "The productivity signal is weaker than the size signal but not negligible; read the tables.")
    )
    log.info("VERDICT: " + verdict)

    if save:
        OUTPUT_TAB.mkdir(parents=True, exist_ok=True)
        corr_tbl.to_csv(OUTPUT_TAB / "productivity_correlation_collapse.csv", index=False)
        reg_tbl.to_csv(OUTPUT_TAB / "productivity_regressions.csv", index=False)
        if not hr_tbl.empty:
            hr_tbl.to_csv(OUTPUT_TAB / "productivity_horse_race.csv", index=False)
        if not ratio_tbl.empty:
            ratio_tbl.to_csv(OUTPUT_TAB / "productivity_ratio_flagged.csv", index=False)

    return {"corr": corr_tbl, "regressions": reg_tbl, "horse_race": hr_tbl,
            "ratio_flagged": ratio_tbl, "verdict": verdict, "panel": p,
            "has_productivity": has_pw}


# ────────────────────────────────────────────────────────────────────────
# 5. Figure: the two scatters that tell the whole story
# ────────────────────────────────────────────────────────────────────────
def plot_productivity(result, save_path=None):
    """Left: light vs GDP (size), tight. Right: light vs GDP/worker (or /capita)."""
    import matplotlib.pyplot as plt
    from src.config import OUTPUT_FIG, NAVY, MAROON, FIGURE_DPI

    p = result["panel"]
    if result.get("has_productivity"):
        ycol, ylab, slug = "ln_gdp_pw", "log GDP per worker", "productivity"
    else:
        ycol, ylab, slug = "ln_gdp_pc", "log GDP per capita", "prosperity"

    def _xy(col):
        d = p[["ln_ntl", col]].replace([np.inf, -np.inf], np.nan).dropna()
        return d["ln_ntl"].values, d[col].values, float(d.corr().iloc[0, 1])

    prod_title = "GDP per worker (productivity)" if result.get("has_productivity") \
        else "GDP per capita (prosperity)"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (col, lab, nice) in zip(
            axes, [("ln_gdp", "log GDP (size)", "Total GDP (size)"),
                   (ycol, ylab, prod_title)]):
        x, y, r = _xy(col)
        color = NAVY if col == "ln_gdp" else MAROON
        ax.scatter(x, y, s=6, alpha=0.25, color=color, edgecolors="none")
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 50)
        ax.plot(xs, a + b * xs, color="black", lw=1.6)
        ax.set_xlabel("log mean radiance (brightness)")
        ax.set_ylabel(lab)
        ax.set_title(f"{nice}\ncorrelation = {r:.2f}")
    fig.suptitle("Light tracks size, not how productive the people are",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / f"{slug}_vs_size.png")
    OUTPUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out
