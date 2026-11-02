"""
The 2020 shock as a cross-sectional test of whether light tracks changes.

The paper argues that light carries little information about short-run GDP
movements. The pandemic is the ideal place to test that, because it produced a
large, sharply varying shock across counties in a single year, followed by a
rebound the next. If light tracks changes, the counties that lost the most output
in 2020 should also have dimmed the most, and the counties that rebounded hardest
in 2021 should have brightened the most.

For each episode this regresses the one-year change in log GDP on the one-year
change in log light across counties, with state fixed effects and standard errors
clustered by state, and draws a binned scatter (residualized on the state effects
so the picture matches the regression). It also reports the bivariate R-squared,
which is the share of the cross-county variance in the GDP change that light
alone explains. A flat binned scatter with a coefficient near zero would say
light missed the shock; a positive slope would say it caught at least part of it.

The closest county-level nighttime-lights work stops in 2019, so this window is
new ground whichever way the result falls, and the result should be read as it
comes out rather than assumed in advance.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import OUTPUT_TAB

log = logging.getLogger(__name__)

EPISODES = [("2020 downturn", 2019, 2020), ("2021 rebound", 2020, 2021)]


def _episode_frame(panel, y0, y1):
    """One row per county with the change in log GDP and log light from y0 to y1."""
    a = panel[panel["year"] == y0][["fips", "state_fips", "ln_gdp", "ln_ntl"]]
    a = a.rename(columns={"ln_gdp": "g0", "ln_ntl": "n0"})
    b = panel[panel["year"] == y1][["fips", "ln_gdp", "ln_ntl"]]
    b = b.rename(columns={"ln_gdp": "g1", "ln_ntl": "n1"})
    m = a.merge(b, on="fips", how="inner")
    m["d_gdp"] = m["g1"] - m["g0"]
    m["d_ntl"] = m["n1"] - m["n0"]
    return m.dropna(subset=["d_gdp", "d_ntl", "state_fips"]).reset_index(drop=True)


def _residualize(y, state):
    """Residuals of y on state dummies, with the overall mean added back."""
    sd = pd.get_dummies(state, prefix="st", drop_first=True, dtype=float)
    X = sm.add_constant(sd)
    fit = sm.OLS(np.asarray(y, float), X).fit()
    return fit.resid + np.mean(y)


def _fit_episode(m):
    """State-FE regression of d_gdp on d_ntl, clustered by state, plus bivariate R2."""
    sd = pd.get_dummies(m["state_fips"], prefix="st", drop_first=True, dtype=float)
    X = pd.DataFrame({"d_ntl": m["d_ntl"].values})
    for c in sd.columns:
        X[c] = sd[c].values
    X = sm.add_constant(X)
    res = sm.OLS(m["d_gdp"].values, X).fit(
        cov_type="cluster", cov_kwds={"groups": m["state_fips"].values})
    # Bivariate R-squared: how much of the GDP-change variance light alone explains.
    biv = sm.OLS(m["d_gdp"].values,
                 sm.add_constant(m[["d_ntl"]].values)).fit()
    return {
        "coef": float(res.params["d_ntl"]),
        "se": float(res.bse["d_ntl"]),
        "pval": float(res.pvalues["d_ntl"]),
        "bivariate_r2": float(biv.rsquared),
        "n": int(res.nobs),
        "mean_d_gdp": float(m["d_gdp"].mean()),
        "mean_d_ntl": float(m["d_ntl"].mean()),
    }


def run_pandemic_test(panel, save=True):
    """Cross-sectional light-GDP relationship during the 2020 shock and 2021 rebound."""
    rows = []
    frames = {}
    for label, y0, y1 in EPISODES:
        if not {y0, y1}.issubset(set(panel["year"])):
            log.warning(f"{label}: years {y0}/{y1} not both in panel, skipped.")
            continue
        m = _episode_frame(panel, y0, y1)
        frames[label] = m
        r = _fit_episode(m)
        r["episode"] = label
        rows.append(r)
        log.info(f"{label}: d_gdp on d_ntl (state FE) = {r['coef']:.4f} "
                 f"(SE {r['se']:.4f}, p {r['pval']:.3f}), bivariate R2 = "
                 f"{r['bivariate_r2']:.4f}, mean d_gdp = {r['mean_d_gdp']:.4f}, "
                 f"mean d_ntl = {r['mean_d_ntl']:.4f}, n = {r['n']}")

    table = pd.DataFrame(rows, columns=["episode", "coef", "se", "pval",
                                        "bivariate_r2", "n", "mean_d_gdp",
                                        "mean_d_ntl"])
    if save and not table.empty:
        table.to_csv(OUTPUT_TAB / "pandemic_test.csv", index=False)
    return {"table": table, "frames": frames}


def plot_pandemic_test(result, nbins=20, save_path=None):
    """Binned scatter of GDP change against light change for each episode."""
    import matplotlib.pyplot as plt
    from src.config import OUTPUT_FIG, NAVY, MAROON

    frames = result["frames"]
    tbl = result["table"].set_index("episode")
    if not frames:
        return None

    fig, axes = plt.subplots(1, len(frames), figsize=(6.5 * len(frames), 5.2),
                             squeeze=False)
    for ax, (label, m) in zip(axes[0], frames.items()):
        rx = _residualize(m["d_ntl"].values, m["state_fips"])
        ry = _residualize(m["d_gdp"].values, m["state_fips"])
        nb = min(nbins, int(np.unique(rx).size))
        codes = pd.qcut(rx, nb, labels=False, duplicates="drop")
        bx = pd.Series(rx).groupby(codes).mean()
        by = pd.Series(ry).groupby(codes).mean()
        ax.scatter(bx, by, color=NAVY, s=40, zorder=3)
        # Overlay the state-FE regression slope through the residual means.
        b = tbl.loc[label, "coef"]
        xs = np.linspace(np.min(rx), np.max(rx), 50)
        ax.plot(xs, np.mean(ry) + b * (xs - np.mean(rx)), color=MAROON, lw=2,
                label=f"slope = {b:.3f} (p = {tbl.loc[label,'pval']:.2f})")
        ax.axhline(np.mean(ry), color="grey", lw=0.6, ls=":")
        ax.set_xlabel("Change in log light (residualized)")
        ax.set_ylabel("Change in log GDP (residualized)")
        ax.set_title(label)
        ax.legend(fontsize=9)
    fig.suptitle("Did counties that lost output also go dark?\n"
                 "County GDP change vs light change, state fixed effects",
                 fontsize=12)
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / "pandemic_test.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out
