"""
Long-difference test for measurement-error attenuation (Griliches & Hausman, 1986).

The within-county light-GDP elasticity is small, but a small coefficient can mean
two very different things: the relationship is genuinely weak, or light measures
GDP with so much noise that fixed effects attenuate a real relationship toward
zero. A single regression cannot tell these apart. Long differences can.

Under classical measurement error, the attenuation in a k-year difference depends
on the signal-to-noise ratio at that horizon. If the measurement error is roughly
white noise, its variance in a k-year difference does not grow with k, while the
variance of the true k-year change in light does grow with k. So the reliability
ratio rises with the horizon, and the estimated coefficient should climb toward
the true elasticity as k increases. A coefficient that rises with k is the
signature of attenuation hiding a real signal; a coefficient that stays flat says
the within relationship is simply weak.

This regresses the k-year change in log GDP on the k-year change in log light, for
k = 1 to 6, with end-year fixed effects and standard errors clustered by county,
and plots the coefficient against k. At k = 1 it is the first difference (with
year effects); the question is what happens as k grows. Note that the panel is
only seven years, so the k = 5 and k = 6 estimates rest on one or two differences
per county and are noisy; the informative range is roughly k = 1 to 4.

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import OUTPUT_TAB

log = logging.getLogger(__name__)


def _long_diff_reg(panel, k):
    """Regress the k-year change in log GDP on the k-year change in log light."""
    d = panel.sort_values(["fips", "year"]).copy()
    d["dgdp"] = d.groupby("fips")["ln_gdp"].diff(k)
    d["dntl"] = d.groupby("fips")["ln_ntl"].diff(k)
    d = d.dropna(subset=["dgdp", "dntl"])
    if d.empty or d["dntl"].std() == 0:
        return None

    # End-year fixed effects. With a single end year (k = 6) the dummy set is
    # empty and the regression keeps only a constant and the light change.
    yd = pd.get_dummies(d["year"], prefix="endyr", drop_first=True, dtype=float)
    X = pd.DataFrame({"dntl": d["dntl"].values})
    for c in yd.columns:
        X[c] = yd[c].values
    X = sm.add_constant(X)
    res = sm.OLS(d["dgdp"].values, X).fit(
        cov_type="cluster", cov_kwds={"groups": d["fips"].values})
    return {
        "k": k,
        "coef": float(res.params["dntl"]),
        "se": float(res.bse["dntl"]),
        "ci_lo": float(res.params["dntl"] - 1.96 * res.bse["dntl"]),
        "ci_hi": float(res.params["dntl"] + 1.96 * res.bse["dntl"]),
        "n": int(res.nobs),
        "n_counties": int(d["fips"].nunique()),
    }


def run_long_differences(panel, kmax=6, save=True):
    """
    Estimate the light-GDP coefficient at horizons k = 1 to kmax and report the
    pattern. A rising coefficient points to attenuation; a flat one points to a
    genuinely weak within relationship.
    """
    rows = []
    for k in range(1, kmax + 1):
        r = _long_diff_reg(panel, k)
        if r:
            rows.append(r)
            log.info(f"k={k}: beta={r['coef']:.4f} (SE={r['se']:.4f}), "
                     f"n={r['n']}, counties={r['n_counties']}")
    table = pd.DataFrame(rows)
    if table.empty:
        log.warning("Long-difference test produced no estimates.")
        return table

    # A simple read of the slope across horizons.
    if len(table) >= 2:
        trend = np.polyfit(table["k"], table["coef"], 1)[0]
        verdict = ("rises with the horizon, consistent with measurement-error "
                   "attenuation hiding a real signal" if trend > 0.005 else
                   "is roughly flat across horizons, consistent with a genuinely "
                   "weak within-county relationship")
        log.info(f"Coefficient {verdict} (slope per added year = {trend:.4f}).")

    if save:
        table.to_csv(OUTPUT_TAB / "long_differences.csv", index=False)
    return table


def plot_long_differences(table, save_path=None):
    """Coefficient against horizon k, with 95 percent confidence intervals."""
    import matplotlib.pyplot as plt
    from src.config import OUTPUT_FIG, NAVY

    fig, ax = plt.subplots(figsize=(8, 5.5))
    yerr = [table["coef"] - table["ci_lo"], table["ci_hi"] - table["coef"]]
    ax.errorbar(table["k"], table["coef"], yerr=yerr, fmt="o-", color=NAVY,
                capsize=4, lw=2, markersize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Difference horizon k (years)")
    ax.set_ylabel("Light-GDP coefficient")
    ax.set_title("Attenuation test: does the elasticity grow with the horizon?\n"
                 "k-year change in log GDP on k-year change in log light")
    ax.set_xticks(table["k"])
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / "long_differences.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out
