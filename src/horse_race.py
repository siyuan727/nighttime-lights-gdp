"""
Does nighttime light improve GDP prediction beyond population?

For a proxy to earn its place, lights must add predictive value over something
cheap you already know: population. This runs a formal out-of-sample horse race
of GDP ~ population against GDP ~ population + lights, two ways:

  (A) LEVELS, across counties. GroupKFold by county, so a county is wholly in
      train or test and we measure transfer to counties never seen. Year
      effects included. This is the "proxy for the size of a place" question.

  (B) CHANGES, within counties. Same CV, on year-to-year growth. This is the
      "proxy for how a place is changing" question. Note population is nearly
      constant here (Census county population is capped at 2019 and reused),
      so the growth baseline is weak by construction, which is worth stating.

Reports out-of-sample R2 and RMSE for the baseline and the full model, and the
improvement from adding lights. The honest, expected pattern given the rest of
the project: lights add real value in levels, little in changes.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

from src.config import OUTPUT_TAB

log = logging.getLogger(__name__)


def _oos(X_base, X_full, y, groups, n_splits):
    """Pooled out-of-sample predictions for baseline and full feature sets."""
    bp, fp = np.zeros_like(y, dtype=float), np.zeros_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(X_base, y, groups):
        bp[te] = LinearRegression().fit(X_base[tr], y[tr]).predict(X_base[te])
        fp[te] = LinearRegression().fit(X_full[tr], y[tr]).predict(X_full[te])
    b_rmse = np.sqrt(mean_squared_error(y, bp))
    f_rmse = np.sqrt(mean_squared_error(y, fp))
    return {
        "base_r2": r2_score(y, bp), "full_r2": r2_score(y, fp),
        "base_rmse": b_rmse, "full_rmse": f_rmse,
        "delta_r2": r2_score(y, fp) - r2_score(y, bp),
        "rmse_reduction_pct": (b_rmse - f_rmse) / b_rmse * 100,
        "n": int(len(y)),
    }


def run_horse_race(panel, n_splits=5, save=True):
    results = {}

    # ---- (A) LEVELS: predict ln(GDP) for unseen counties ----
    df = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ln_gdp", "ln_pop", "ln_ntl", "fips", "year"]).copy()
    yd = pd.get_dummies(df["year"], prefix="yr", drop_first=True, dtype=float)
    X_base = pd.concat([df[["ln_pop"]], yd], axis=1).values
    X_full = pd.concat([df[["ln_pop", "ln_ntl"]], yd], axis=1).values
    results["levels"] = _oos(X_base, X_full, df["ln_gdp"].values,
                             df["fips"].values, n_splits)

    # ---- (B) CHANGES: predict GDP growth, beyond population growth ----
    g = panel.replace([np.inf, -np.inf], np.nan).sort_values(
        ["fips", "year"]).copy()
    g["pop_growth"] = g.groupby("fips")["ln_pop"].diff()
    g = g.dropna(subset=["gdp_growth", "ntl_growth", "pop_growth", "fips"])
    Xb = g[["pop_growth"]].values
    Xf = g[["pop_growth", "ntl_growth"]].values
    results["changes"] = _oos(Xb, Xf, g["gdp_growth"].values,
                              g["fips"].values, n_splits)

    rows = [{
        "test": k,
        "baseline_R2": round(r["base_r2"], 4),
        "plus_lights_R2": round(r["full_r2"], 4),
        "delta_R2": round(r["delta_r2"], 4),
        "RMSE_reduction_%": round(r["rmse_reduction_pct"], 2),
        "n": r["n"],
    } for k, r in results.items()]
    table = pd.DataFrame(rows).set_index("test")

    log.info("\nLight-vs-population horse race (out-of-sample, GroupKFold by "
             "county):\n" + table.to_string())
    log.info("Reading it: baseline = population only; plus_lights adds ln(NTL). "
             "delta_R2 and RMSE reduction are what lights buy beyond population.")

    if save:
        table.to_csv(OUTPUT_TAB / "horse_race.csv")
    return results


def plot_horse_race(results, save_path=None):
    import matplotlib.pyplot as plt
    from src.config import OUTPUT_FIG, NAVY, AMBER

    tests = ["levels", "changes"]
    labels = ["Levels\n(GDP size across counties)",
              "Changes\n(GDP growth within counties)"]
    base = [results[t]["base_r2"] for t in tests]
    full = [results[t]["full_r2"] for t in tests]

    x = np.arange(len(tests))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, base, w, label="Population only", color=NAVY)
    ax.bar(x + w / 2, full, w, label="Population + lights", color=AMBER)
    for i, t in enumerate(tests):
        ax.annotate(f"+{results[t]['delta_r2']:.3f}",
                    (x[i] + w / 2, full[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Out-of-sample R-squared")
    ax.set_title("Does light add anything beyond population?\n"
                 "Out-of-sample GDP prediction, GroupKFold by county")
    ax.legend()
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / "horse_race.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out
