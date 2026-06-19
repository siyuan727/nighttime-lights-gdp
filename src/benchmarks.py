import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from src.config import OUTPUT_TAB

log = logging.getLogger(__name__)


def run_pooled_ols(panel):
    results = {}
    Y = panel["ln_gdp"]

    X1 = sm.add_constant(panel["ln_ntl"])
    results["ols_bivariate"] = sm.OLS(Y, X1).fit(
        cov_type="cluster", cov_kwds={"groups": panel["fips"]})

    X2 = sm.add_constant(panel[["ln_ntl","ln_pop"]])
    results["ols_pop"] = sm.OLS(Y, X2).fit(
        cov_type="cluster", cov_kwds={"groups": panel["fips"]})

    state_dummies = pd.get_dummies(
        panel["state_fips"], prefix="state", drop_first=True, dtype=float)
    X3 = sm.add_constant(pd.concat([panel[["ln_ntl","ln_pop"]], state_dummies], axis=1))
    results["ols_state_fe"] = sm.OLS(Y, X3).fit(
        cov_type="cluster", cov_kwds={"groups": panel["fips"]})

    diff = panel.dropna(subset=["gdp_growth","ntl_growth"])
    X4   = sm.add_constant(diff["ntl_growth"])
    results["ols_fd"] = sm.OLS(diff["gdp_growth"], X4).fit(
        cov_type="cluster", cov_kwds={"groups": diff["fips"]})

    rows = []
    for name, res in results.items():
        var = "ntl_growth" if name == "ols_fd" else "ln_ntl"
        if var in res.params.index:
            rows.append({"model": name,
                         "coef": round(res.params[var], 4),
                         "se":   round(res.bse[var], 4),
                         "pval": round(res.pvalues[var], 4),
                         "r2":   round(res.rsquared, 4),
                         "nobs": int(res.nobs)})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(OUTPUT_TAB / "ols_benchmarks.csv")
    log.info("\nOLS Benchmarks:\n" + table.to_string())
    return results