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

INFERENCE (run_horse_race_inference). A point gain in R-squared is not enough
for a referee; they want to know whether it is distinguishable from zero. Two
tools, both clustered by county so neighboring observations are not treated as
independent:

  - a cluster (block) bootstrap over counties, giving a standard error and a
    95 percent confidence interval for the gain in out-of-sample R-squared;
  - a Diebold-Mariano test on the per-observation squared-error loss
    differential (baseline minus full), clustered by county, testing the
    one-sided null that lights do not reduce prediction error.

The expected result: the levels gain is large and highly significant, and the
changes gain is tiny and, in all likelihood, not significant, which is a
cleaner and stronger way to say "lights do not track short-run changes."
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

from src.config import OUTPUT_TAB

log = logging.getLogger(__name__)


def _oos_predictions(X_base, X_full, y, groups, n_splits):
    """Pooled out-of-sample predictions for baseline and full feature sets."""
    bp = np.zeros_like(y, dtype=float)
    fp = np.zeros_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(X_base, y, groups):
        bp[te] = LinearRegression().fit(X_base[tr], y[tr]).predict(X_base[te])
        fp[te] = LinearRegression().fit(X_full[tr], y[tr]).predict(X_full[te])
    return bp, fp


def _oos(X_base, X_full, y, groups, n_splits):
    """Out-of-sample fit statistics plus the raw predictions for inference."""
    bp, fp = _oos_predictions(X_base, X_full, y, groups, n_splits)
    b_rmse = np.sqrt(mean_squared_error(y, bp))
    f_rmse = np.sqrt(mean_squared_error(y, fp))
    return {
        "base_r2": r2_score(y, bp), "full_r2": r2_score(y, fp),
        "base_rmse": b_rmse, "full_rmse": f_rmse,
        "delta_r2": r2_score(y, fp) - r2_score(y, bp),
        "rmse_reduction_pct": (b_rmse - f_rmse) / b_rmse * 100,
        "n": int(len(y)),
        "_y": y, "_bp": bp, "_fp": fp, "_groups": np.asarray(groups),
    }


def _design(panel):
    """Build the levels and changes design matrices once, reused everywhere."""
    out = {}
    # Levels.
    df = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ln_gdp", "ln_pop", "ln_ntl", "fips", "year"]).copy()
    yd = pd.get_dummies(df["year"], prefix="yr", drop_first=True, dtype=float)
    out["levels"] = {
        "X_base": pd.concat([df[["ln_pop"]], yd], axis=1).values,
        "X_full": pd.concat([df[["ln_pop", "ln_ntl"]], yd], axis=1).values,
        "y": df["ln_gdp"].values, "groups": df["fips"].values,
    }
    # Changes.
    g = panel.replace([np.inf, -np.inf], np.nan).sort_values(["fips", "year"]).copy()
    g["pop_growth"] = g.groupby("fips")["ln_pop"].diff()
    g = g.dropna(subset=["gdp_growth", "ntl_growth", "pop_growth", "fips"])
    out["changes"] = {
        "X_base": g[["pop_growth"]].values,
        "X_full": g[["pop_growth", "ntl_growth"]].values,
        "y": g["gdp_growth"].values, "groups": g["fips"].values,
    }
    return out


def run_horse_race(panel, n_splits=5, save=True):
    design = _design(panel)
    results = {}
    for key in ("levels", "changes"):
        d = design[key]
        results[key] = _oos(d["X_base"], d["X_full"], d["y"], d["groups"], n_splits)

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


# ─────────────────────────── inference ───────────────────────────

def _cluster_bootstrap_delta_r2(y, bp, fp, groups, n_boot=2000, seed=42):
    """
    Cluster (block) bootstrap of the out-of-sample gain in R-squared. Counties
    are resampled with replacement; within a draw the delta-R-squared is
    recomputed on the held-out predictions already in hand, so no model is
    refit. Returns the bootstrap SE and percentile 95% CI of delta-R-squared.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}

    def r2(y_, p_):
        ss_res = np.sum((y_ - p_) ** 2)
        ss_tot = np.sum((y_ - y_.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(uniq, size=uniq.size, replace=True)
        take = np.concatenate([idx_by_g[g] for g in draw])
        yy, bb, ff = y[take], bp[take], fp[take]
        deltas[b] = r2(yy, ff) - r2(yy, bb)

    deltas = deltas[np.isfinite(deltas)]
    return {
        "delta_r2_boot_mean": float(np.mean(deltas)),
        "delta_r2_se": float(np.std(deltas, ddof=1)),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "n_boot": int(deltas.size),
    }


def _diebold_mariano_clustered(y, bp, fp, groups):
    """
    One-sided Diebold-Mariano test that the full model has lower squared-error
    loss than the baseline, using the loss differential d = e_base^2 - e_full^2.
    Positive mean d means lights help. The standard error of mean d is clustered
    by county (cluster-robust), which is the right unit of dependence here.
    """
    d = (y - bp) ** 2 - (y - fp) ** 2
    dbar = float(np.mean(d))
    n = d.size

    uniq = np.unique(groups)
    G = uniq.size
    # Cluster-robust variance of the mean of d: treat clusters as independent.
    # Var(mean d) = (G/(G-1)) * sum_g (S_g - n_g * dbar)^2 / n^2, where S_g is
    # the within-cluster sum and n_g the cluster size.
    S_g = np.array([d[groups == g].sum() for g in uniq])
    n_g = np.array([(groups == g).sum() for g in uniq], dtype=float)
    adj = G / (G - 1.0) if G > 1 else 1.0
    var_mean = adj * np.sum((S_g - n_g * dbar) ** 2) / (n ** 2)
    se_mean = np.sqrt(var_mean) if var_mean > 0 else np.nan
    dm = dbar / se_mean if (se_mean and se_mean > 0) else np.nan

    from scipy import stats
    p_one = float(stats.t.sf(dm, df=max(G - 1, 1))) if np.isfinite(dm) else np.nan
    return {
        "mean_loss_diff": dbar,
        "se_clustered": float(se_mean) if np.isfinite(se_mean) else np.nan,
        "dm_stat": float(dm) if np.isfinite(dm) else np.nan,
        "p_one_sided": p_one,
        "n_clusters": int(G),
    }


def run_horse_race_inference(panel, n_splits=5, n_boot=2000, save=True, seed=42):
    """
    Attach uncertainty to the horse-race gains: a cluster-bootstrap CI on the
    gain in out-of-sample R-squared and a county-clustered Diebold-Mariano test
    of whether lights significantly reduce prediction error, for both the
    levels and the changes tasks.
    """
    design = _design(panel)
    rows = []
    for key in ("levels", "changes"):
        d = design[key]
        bp, fp = _oos_predictions(d["X_base"], d["X_full"], d["y"],
                                  d["groups"], n_splits)
        y, groups = d["y"], np.asarray(d["groups"])

        boot = _cluster_bootstrap_delta_r2(y, bp, fp, groups,
                                           n_boot=n_boot, seed=seed)
        dm = _diebold_mariano_clustered(y, bp, fp, groups)
        delta_point = r2_score(y, fp) - r2_score(y, bp)

        rows.append({
            "test": key,
            "delta_R2": round(float(delta_point), 4),
            "delta_R2_se": round(boot["delta_r2_se"], 4),
            "ci_lo": round(boot["ci_lo"], 4),
            "ci_hi": round(boot["ci_hi"], 4),
            "DM_stat": (round(dm["dm_stat"], 3)
                        if dm["dm_stat"] == dm["dm_stat"] else np.nan),
            "DM_p_one_sided": (round(dm["p_one_sided"], 4)
                               if dm["p_one_sided"] == dm["p_one_sided"] else np.nan),
            "n_clusters": dm["n_clusters"],
            "n": int(len(y)),
        })
        log.info(f"[{key}] delta R2 = {delta_point:.4f} "
                 f"[95% CI {boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}], "
                 f"DM = {dm['dm_stat']:.2f}, one-sided p = {dm['p_one_sided']:.4f} "
                 f"({dm['n_clusters']} county clusters)")

    table = pd.DataFrame(rows).set_index("test")
    log.info("\nHorse-race inference (cluster bootstrap + Diebold-Mariano):\n"
             + table.to_string())
    log.info("Read it: the levels gain should be large with a CI well above "
             "zero; the changes gain should be small with a CI that includes or "
             "sits near zero and a non-significant DM test. That is the formal "
             "version of 'lights price the level, not the change'.")

    if save:
        table.to_csv(OUTPUT_TAB / "horse_race_inference.csv")
    return table


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
