"""
Separating attenuation from omitted-variable bias.

The paper's central number is the collapse of the light-GDP elasticity from
about 0.845 (pooled) to about 0.162 (two-way fixed effects). Two forces drive
that collapse and a single regression cannot tell them apart:

  1. Omitted-variable bias / composition. The pooled estimate credits light with
     the effect of fixed, place-specific factors that make a county both bright
     and large. Fixed effects remove this.
  2. Attenuation. Light measures activity with error. Fixed effects keep only
     the within-county variation, which is the noisiest part, so the signal-to-
     noise ratio falls and the coefficient is biased further toward zero.

To measure attenuation instead of asserting it, use two independent noisy
measures of the same latent annual brightness for each county: the mean radiance
over the first half of the year (ln_ntl_a) and over the second half (ln_ntl_b).
Under classical measurement error with errors that are uncorrelated across the
two halves, instrumenting one with the other in the within regression recovers
the attenuation-corrected elasticity (errors-in-variables IV). The estimator is
just-identified, so

    beta_IV = (z . y) / (z . x)        on two-way-demeaned y, x = ln_ntl_a, z = ln_ntl_b.

The reliability ratio is then beta_FE_naive / beta_IV (a number in (0, 1]; lower
means more attenuation), and the collapse decomposes as

    pooled - FE_naive   =   (pooled - beta_IV)    +   (beta_IV - FE_naive)
    [total collapse]        [composition / OVB]       [attenuation in the within]

If two half-year measures are unavailable, a fallback instruments contemporaneous
light with its one-year lag. That is weaker, because it is only valid if the
measurement errors are serially uncorrelated, which is a stronger assumption than
independence across two same-year composites, so it is reported as a robustness
variant, not the headline.

Data note: the two half-year measures require the MONTHLY Black Marble product
(VNP46A3) aggregated to counties, which is a heavier download than the annual
VNP46A4 used elsewhere. get_county_ntl_split builds them and caches the result.
"""

import logging

import numpy as np
import pandas as pd

from src.config import OUTPUT_TAB

log = logging.getLogger(__name__)


# ── Two-way within (fixed-effects) transform ─────────────────

def _twoway_demean(df, cols, entity="fips", time="year", iterations=20, tol=1e-10):
    """
    Subtract entity and time means from each column (two-way within transform)
    by alternating projections, which converges for unbalanced panels too.
    Returns a copy of df with the demeaned columns replaced.
    """
    out = df[cols].astype(float).copy()
    ent = df[entity].values
    tim = df[time].values
    for _ in range(iterations):
        before = out.to_numpy().copy()
        for g in (ent, tim):
            out = out.groupby(g, sort=False).transform(lambda s: s - s.mean())
        if np.nanmax(np.abs(out.to_numpy() - before)) < tol:
            break
    return out


# ── Just-identified IV with cluster-robust SE ────────────────

def _iv_just_identified(y, x, z, clusters):
    """
    beta for y on x, instrumenting x with z (all 1-D, already demeaned), with
    standard errors clustered by `clusters`. Also returns the first-stage F
    (x on z) as an instrument-strength diagnostic.
    """
    y = np.asarray(y, float); x = np.asarray(x, float); z = np.asarray(z, float)
    zx = float(z @ x)
    beta = float((z @ y) / zx)
    resid = y - x * beta

    # Cluster-robust variance: (Z'X)^-1 [ sum_g (z_g'e_g)^2 ] (X'Z)^-1.
    uniq = np.unique(clusters)
    G = uniq.size
    meat = 0.0
    for g in uniq:
        m = clusters == g
        s = float(z[m] @ resid[m])
        meat += s * s
    var = meat / (zx * zx)
    adj = G / (G - 1.0) if G > 1 else 1.0
    se = np.sqrt(adj * var)

    # First-stage: x on z (through the origin, data are demeaned).
    phi = float((z @ x) / (z @ z))
    xhat = z * phi
    ss_res = float(((x - xhat) ** 2).sum())
    ss_tot = float((x ** 2).sum())
    r2_fs = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n = y.size
    f_stat = (r2_fs / (1.0 - r2_fs)) * (n - 1) if r2_fs < 1 else np.inf

    return {"beta": beta, "se": float(se), "first_stage_F": float(f_stat),
            "n": int(n), "n_clusters": int(G)}


def _ols_within(y, x, clusters):
    """Naive within (FE) slope of y on x, 1-D demeaned, cluster-robust SE."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    xx = float(x @ x)
    beta = float((x @ y) / xx)
    resid = y - x * beta
    uniq = np.unique(clusters); G = uniq.size
    meat = sum(float(x[clusters == g] @ resid[clusters == g]) ** 2 for g in uniq)
    adj = G / (G - 1.0) if G > 1 else 1.0
    se = np.sqrt(adj * meat / (xx * xx))
    return {"beta": beta, "se": float(se), "n": int(y.size), "n_clusters": int(G)}


def estimate_attenuation(panel, measure_a="ln_ntl_a", measure_b="ln_ntl_b",
                         beta_pooled=None, save=True):
    """
    Attenuation-corrected within elasticity via errors-in-variables IV using two
    half-year light measures. `panel` must have ln_gdp, the two measures, fips,
    year. beta_pooled (e.g. 0.845 from the OLS benchmark) is optional and only
    used to print the full pooled -> within decomposition.
    """
    need = ["ln_gdp", measure_a, measure_b, "fips", "year"]
    d = panel.dropna(subset=need).copy()
    dem = _twoway_demean(d, ["ln_gdp", measure_a, measure_b])
    y = dem["ln_gdp"].values
    xa = dem[measure_a].values
    xb = dem[measure_b].values
    cl = d["fips"].values

    naive = _ols_within(y, xa, cl)               # FE using measure A
    iv = _iv_just_identified(y, xa, xb, cl)       # A instrumented by B

    reliability = naive["beta"] / iv["beta"] if iv["beta"] else np.nan
    log.info(f"Attenuation IV: FE_naive beta = {naive['beta']:.4f} "
             f"(SE {naive['se']:.4f}), IV beta = {iv['beta']:.4f} "
             f"(SE {iv['se']:.4f}), first-stage F = {iv['first_stage_F']:.1f}, "
             f"reliability ratio = {reliability:.3f}")

    rows = [
        {"quantity": "FE within (naive, measure A)", "value": round(naive["beta"], 4),
         "se": round(naive["se"], 4)},
        {"quantity": "IV within (attenuation-corrected)", "value": round(iv["beta"], 4),
         "se": round(iv["se"], 4)},
        {"quantity": "reliability ratio (FE/IV)", "value": round(float(reliability), 4),
         "se": np.nan},
        {"quantity": "first-stage F", "value": round(iv["first_stage_F"], 2), "se": np.nan},
    ]
    if beta_pooled is not None:
        total = beta_pooled - naive["beta"]
        attenuation = iv["beta"] - naive["beta"]
        composition = beta_pooled - iv["beta"]
        rows += [
            {"quantity": "pooled OLS beta", "value": round(float(beta_pooled), 4), "se": np.nan},
            {"quantity": "total collapse (pooled - FE)", "value": round(total, 4), "se": np.nan},
            {"quantity": "  of which composition/OVB (pooled - IV)",
             "value": round(composition, 4), "se": np.nan},
            {"quantity": "  of which attenuation (IV - FE)",
             "value": round(attenuation, 4), "se": np.nan},
        ]
        log.info(f"Decomposition of the collapse: total {total:.3f} = "
                 f"composition {composition:.3f} + attenuation {attenuation:.3f}")

    table = pd.DataFrame(rows)
    if save:
        table.to_csv(OUTPUT_TAB / "measurement_error.csv", index=False)
    return {"naive": naive, "iv": iv, "reliability": float(reliability),
            "table": table}


def estimate_attenuation_lag_fallback(panel, beta_pooled=None, save=True):
    """
    Fallback when two half-year measures are not available: instrument
    contemporaneous ln_ntl with its one-year lag. Valid only if measurement
    errors are serially uncorrelated (a stronger assumption than two same-year
    composites), so treat it as a robustness variant.
    """
    d = panel.sort_values(["fips", "year"]).copy()
    d["ln_ntl_lag"] = d.groupby("fips")["ln_ntl"].shift(1)
    d = d.dropna(subset=["ln_gdp", "ln_ntl", "ln_ntl_lag", "fips", "year"])
    dem = _twoway_demean(d, ["ln_gdp", "ln_ntl", "ln_ntl_lag"])
    y = dem["ln_gdp"].values
    x = dem["ln_ntl"].values
    z = dem["ln_ntl_lag"].values
    cl = d["fips"].values

    naive = _ols_within(y, x, cl)
    iv = _iv_just_identified(y, x, z, cl)
    reliability = naive["beta"] / iv["beta"] if iv["beta"] else np.nan
    log.info(f"Attenuation IV (lag instrument): FE_naive = {naive['beta']:.4f}, "
             f"IV = {iv['beta']:.4f} (SE {iv['se']:.4f}), "
             f"first-stage F = {iv['first_stage_F']:.1f}, "
             f"reliability = {reliability:.3f}")

    rows = [
        {"quantity": "FE within (naive)", "value": round(naive["beta"], 4), "se": round(naive["se"], 4)},
        {"quantity": "IV within (lag instrument)", "value": round(iv["beta"], 4), "se": round(iv["se"], 4)},
        {"quantity": "reliability ratio (FE/IV)", "value": round(float(reliability), 4), "se": np.nan},
        {"quantity": "first-stage F", "value": round(iv["first_stage_F"], 2), "se": np.nan},
    ]
    table = pd.DataFrame(rows)
    if save:
        table.to_csv(OUTPUT_TAB / "measurement_error_lag.csv", index=False)
    return {"naive": naive, "iv": iv, "reliability": float(reliability), "table": table}


# ── Data: two half-year light measures from monthly Black Marble ──

def get_county_ntl_split(years=None, use_cache=True):
    """
    Two independent half-year radiance measures per county-year, from the MONTHLY
    Black Marble product (VNP46A3): ntl_h1 is the mean of January-June, ntl_h2
    the mean of July-December. Returns [fips, year, ntl_h1, ntl_h2]. Caches to
    CSV. This is a heavier download than the annual product, since it pulls the
    US tiles for twelve months per year instead of one.
    """
    from datetime import date
    from src.config import DATA_RAW, YEARS, NASA_TOKEN_ENV
    from src.ntl_counties_bm import _county_gdf, _token

    cache = DATA_RAW / "ntl_county_split.csv"
    years = list(years or YEARS)
    if use_cache and cache.exists():
        log.info(f"Loading cached split NTL from {cache.name}")
        return pd.read_csv(cache, dtype={"fips": str})

    from blackmarble.extract import bm_extract
    from blackmarble.types import Product

    gdf = _county_gdf()
    token = _token()
    bm_dir = DATA_RAW / "blackmarble_us_monthly"
    bm_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for y in years:
        months = [date(y, m, 1) for m in range(1, 13)]
        try:
            res = bm_extract(gdf, Product.VNP46A3, months, token,
                             aggfunc=["mean"], output_directory=bm_dir,
                             check_all_tiles_exist=False)
        except Exception as e:
            log.warning(f"Monthly Black Marble {y} failed: {e}")
            continue
        m = pd.DataFrame(res)
        # Resolve fips, date, and the mean-radiance column defensively.
        fips_col = next((c for c in m.columns if c.lower() == "fips"), None)
        date_col = next((c for c in m.columns if c.lower() in ("date", "time")), None)
        mean_col = next((c for c in m.columns
                         if "mean" in c.lower() and c not in (fips_col, date_col)), None)
        if mean_col is None:
            mean_col = next((c for c in m.columns
                             if pd.api.types.is_numeric_dtype(m[c])
                             and c not in (fips_col, date_col)), None)
        m["month"] = pd.to_datetime(m[date_col].astype(str), errors="coerce").dt.month
        m["fips"] = m[fips_col].astype(str).str.zfill(5)
        m["half"] = np.where(m["month"] <= 6, "h1", "h2")
        agg = (m.groupby(["fips", "half"])[mean_col].mean()
               .unstack("half").reset_index())
        agg["year"] = y
        agg = agg.rename(columns={"h1": "ntl_h1", "h2": "ntl_h2"})
        frames.append(agg[["fips", "year", "ntl_h1", "ntl_h2"]])
        log.info(f"Monthly NTL {y}: {len(agg)} counties.")

    if not frames:
        raise RuntimeError("Monthly Black Marble returned no data.")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cache, index=False)
    log.info(f"Cached split NTL: {len(out)} rows, {out['fips'].nunique()} counties.")
    return out


def attach_split_measures(panel, split_df, offset=0.01):
    """
    Merge the two half-year measures onto the main panel and log-transform them
    with the same offset used for ln_ntl, producing ln_ntl_a and ln_ntl_b.
    """
    s = split_df.copy()
    s["fips"] = s["fips"].astype(str).str.zfill(5)
    m = panel.merge(s, on=["fips", "year"], how="inner")
    m = m[(m["ntl_h1"] > 0) & (m["ntl_h2"] > 0)].copy()
    m["ln_ntl_a"] = np.log(m["ntl_h1"] + offset)
    m["ln_ntl_b"] = np.log(m["ntl_h2"] + offset)
    log.info(f"Attached split measures: {len(m)} county-years with both halves.")
    return m
