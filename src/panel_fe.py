import pandas as pd
import numpy as np
import logging
from src.config import OUTPUT_TAB, OUTPUT_LOG

log = logging.getLogger(__name__)


def _prep(df):
    df = df.copy()
    df["fips"] = df["fips"].astype(str)
    df["year"] = pd.to_datetime(df["year"], format="%Y")
    df = df.dropna(subset=["ln_gdp","ln_ntl","ln_pop"])
    return df.set_index(["fips","year"])


def run_twoway_fe(panel):
    from linearmodels.panel import PanelOLS
    results = {}
    df = _prep(panel)

    for name, entity, time, vars_ in [
        ("fe_entity",     True,  False, ["ln_ntl"]),
        ("fe_twoway",     True,  True,  ["ln_ntl"]),
        ("fe_twoway_pop", True,  True,  ["ln_ntl","ln_pop"]),
    ]:
        log.info(f"Estimating {name}...")
        mod = PanelOLS(df["ln_gdp"], df[vars_],
                       entity_effects=entity, time_effects=time,
                       drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        results[name] = res
        b  = res.params["ln_ntl"]
        se = res.std_errors["ln_ntl"]
        log.info(f"  beta(ln_ntl) = {b:.4f} (SE={se:.4f})")

    _save(results)
    return results

def run_lag_fe(panel):
    """
    Robustness check: two-way FE with LAGGED luminosity, lum_{i,t-1}, plus
    population. Lagging lights reduces simultaneity, since this year's GDP
    cannot drive last year's light. Drops the first sample year (2016), which
    has no prior year to lag from. This is a robustness check on the main
    spec
    """
    from linearmodels.panel import PanelOLS
    df = panel.sort_values(["fips", "year"]).copy()
    df["ln_ntl_lag"] = df.groupby("fips")["ln_ntl"].shift(1)
    df["fips"] = df["fips"].astype(str)
    df["year"] = pd.to_datetime(df["year"], format="%Y")
    df = df.dropna(subset=["ln_gdp", "ln_ntl_lag", "ln_pop"])
    df = df.set_index(["fips", "year"])

    res = PanelOLS(df["ln_gdp"], df[["ln_ntl_lag", "ln_pop"]],
                   entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True)
    b = res.params["ln_ntl_lag"]
    se = res.std_errors["ln_ntl_lag"]
    log.info(f"fe_twoway_lag: beta(ln_ntl_lag) = {b:.4f} (SE={se:.4f}), "
             f"nobs={int(res.nobs)}")
    return {"fe_twoway_lag": res}

def _save(results):
    rows = []
    for name, res in results.items():
        for var in res.params.index:
            rows.append({"model": name, "variable": var,
                         "coef":  round(float(res.params[var]), 4),
                         "se":    round(float(res.std_errors[var]), 4),
                         "pval":  round(float(res.pvalues[var]), 4),
                         "r2":    round(float(res.rsquared), 4),
                         "nobs":  int(res.nobs)})
    pd.DataFrame(rows).to_csv(OUTPUT_TAB / "panel_fe_results.csv", index=False)
    with open(OUTPUT_LOG / "panel_fe_summary.txt", "w") as f:
        for name, res in results.items():
            f.write(f"\n{'='*60}\n{name}\n{'='*60}\n{res.summary}\n")


def run_heterogeneous_fe(panel):
    from linearmodels.panel import PanelOLS
    results = {}
    for label, val in [("urban", 1), ("rural", 0)]:
        sub = panel[panel["urban"] == val].copy()
        if len(sub) < 100:
            continue
        df  = _prep(sub)
        res = PanelOLS(df["ln_gdp"], df[["ln_ntl"]],
                       entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(
            cov_type="clustered", cluster_entity=True)
        results[label] = res
        log.info(f"{label}: beta={res.params['ln_ntl']:.4f} "
                 f"(SE={res.std_errors['ln_ntl']:.4f})")
    return results