"""
Conley (1999) spatial-HAC standard errors.

Clustering by county does nothing about the fact that neighboring counties'
shocks are correlated: a boom in one county spills into the next, so the errors
are correlated across space, not just within a county over time. Ordinary and
county-clustered standard errors ignore that and are too small, which overstates
how precisely the light-GDP relationship is pinned down. Conley standard errors
fix this by letting the error covariance decay with distance out to a cutoff.

This module implements the spatial-HAC sandwich directly:

    V = (X'X)^-1  [ sum_t sum_i sum_j  K(d_ij) * (x_it e_it)(x_jt e_jt)' ]  (X'X)^-1

with a Bartlett kernel K(d) = max(0, 1 - d/cutoff) in great-circle distance.
In a panel, spatial correlation is summed within each year, and serial
correlation within a county across years is added with a Bartlett kernel in
time (lag_years). Set lag_years = 0 for a pure cross-sectional spatial HAC.

Great-circle distances use the haversine formula on county-centroid longitude
and latitude in degrees, so no projection is needed. For ~3,000 counties the
per-year distance matrix is about 70 MB, which is fine; if memory is tight,
lower the cutoff and the kernel is sparser but the loop is the same.

Reference: Conley, T. G. 1999. "GMM Estimation with Cross Sectional
Dependence." Journal of Econometrics 92 (1): 1-45. The panel implementation
follows the spatial-plus-serial form used by Hsiang (2010).
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EARTH_KM = 6371.0


def _haversine_km(lon, lat):
    """Full great-circle distance matrix (km) for points given in degrees."""
    lon = np.radians(np.asarray(lon, float))
    lat = np.radians(np.asarray(lat, float))
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def conley_se(y, X, lon, lat, cutoff_km=500.0, time=None, lag_years=0,
              add_constant=True):
    """
    OLS point estimates with Conley spatial-HAC standard errors.

    Parameters
    ----------
    y : (n,) outcome.
    X : (n, k) regressors WITHOUT a constant unless add_constant=False.
    lon, lat : (n,) county-centroid coordinates in degrees.
    cutoff_km : spatial Bartlett kernel cutoff in kilometers.
    time : (n,) integer year for each observation, or None for a cross-section.
    lag_years : serial-correlation bandwidth in years (Bartlett). 0 = none.
    add_constant : prepend a column of ones.

    Returns
    -------
    dict with params, conley_se, ols_se (classical), t, names, and settings.
    """
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    lon = np.asarray(lon, float).ravel()
    lat = np.asarray(lat, float).ravel()
    n = y.shape[0]

    if add_constant:
        X = np.column_stack([np.ones(n), X])
        names = ["const"] + [f"x{j}" for j in range(1, X.shape[1])]
    else:
        names = [f"x{j}" for j in range(X.shape[1])]
    k = X.shape[1]

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # Classical (homoskedastic) SE for comparison.
    dof = max(n - k, 1)
    sigma2 = (resid @ resid) / dof
    ols_se = np.sqrt(np.diag(sigma2 * XtX_inv))

    # Score contributions u_i = x_i * e_i  (n, k).
    U = X * resid[:, None]

    # Spatial meat, summed within each time period.
    meat = np.zeros((k, k))
    if time is None:
        periods = [np.arange(n)]
    else:
        time = np.asarray(time)
        periods = [np.where(time == t)[0] for t in np.unique(time)]

    for idx in periods:
        if idx.size == 0:
            continue
        D = _haversine_km(lon[idx], lat[idx])
        W = np.clip(1.0 - D / cutoff_km, 0.0, None)   # Bartlett in distance
        Ui = U[idx]
        meat += Ui.T @ W @ Ui

    # Serial correlation within county across years (Bartlett in time).
    if time is not None and lag_years and lag_years > 0:
        # Need a stable unit id; infer from identical (lon,lat) pairs.
        coord_key = np.round(lon, 6).astype(str) + "_" + np.round(lat, 6).astype(str)
        order = np.lexsort((time, coord_key))
        ck = coord_key[order]
        tt = time[order]
        Uo = U[order]
        # For each lag L, add weighted cross-products of same-unit obs L years apart.
        for L in range(1, int(lag_years) + 1):
            w = 1.0 - L / (lag_years + 1.0)        # Bartlett weight
            for a in range(len(order)):
                b = a + L
                if b >= len(order):
                    break
                if ck[a] == ck[b] and (tt[b] - tt[a]) == L:
                    cross = np.outer(Uo[a], Uo[b])
                    meat += w * (cross + cross.T)

    V = XtX_inv @ meat @ XtX_inv
    conley = np.sqrt(np.clip(np.diag(V), 0.0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = beta / conley

    log.info(f"Conley SE: n={n}, k={k}, cutoff={cutoff_km}km, "
             f"lag_years={lag_years}, "
             f"periods={len(periods)}")
    return {
        "names": names,
        "params": beta,
        "conley_se": conley,
        "ols_se": ols_se,
        "t": tstat,
        "n": n,
        "cutoff_km": cutoff_km,
        "lag_years": lag_years,
    }


def centroids_from_shapefile(shp_path, exclude_prefix=None):
    """
    County-centroid [fips, lon, lat] from a TIGER county shapefile. Centroids
    are computed in an equal-area projection then converted back to lon/lat so
    they sit inside the county rather than being distorted near the edges.
    """
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    if "STATEFP" in gdf.columns and "COUNTYFP" in gdf.columns:
        gdf["fips"] = (gdf["STATEFP"].astype(str).str.zfill(2)
                       + gdf["COUNTYFP"].astype(str).str.zfill(3))
    elif "GEOID" in gdf.columns:
        gdf["fips"] = gdf["GEOID"].astype(str).str.zfill(5)
    if exclude_prefix:
        gdf = gdf[~gdf["fips"].str[:2].isin(exclude_prefix)]
    gdf = gdf.to_crs(5070)                      # CONUS Albers equal area
    cent = gdf.geometry.centroid.to_crs(4326)
    out = pd.DataFrame({"fips": gdf["fips"].values,
                        "lon": cent.x.values, "lat": cent.y.values})
    return out.reset_index(drop=True)


def run_conley_levels(panel, shp_path, cutoff_km=500.0, lag_years=1,
                      with_population=True):
    """
    Conley SEs for the pooled levels regression ln_gdp ~ ln_ntl (+ ln_pop),
    with year effects, which is where the strong cross-sectional result lives.
    Attaches county centroids from the shapefile and reports Conley alongside
    the classical SE so the reader sees how much spatial correlation inflates
    the true standard error.
    """
    cent = centroids_from_shapefile(shp_path)
    df = panel.merge(cent, on="fips", how="inner").dropna(
        subset=["ln_gdp", "ln_ntl", "ln_pop", "lon", "lat"])
    log.info(f"Conley levels regression: matched {df['fips'].nunique()} "
             f"counties to centroids, {len(df)} obs.")

    yd = pd.get_dummies(df["year"], prefix="yr", drop_first=True, dtype=float)
    base = [df[["ln_ntl", "ln_pop"]]] if with_population else [df[["ln_ntl"]]]
    X = pd.concat(base + [yd], axis=1)

    res = conley_se(df["ln_gdp"].values, X.values,
                    df["lon"].values, df["lat"].values,
                    cutoff_km=cutoff_km, time=df["year"].values,
                    lag_years=lag_years, add_constant=True)

    # Name the ln_ntl coefficient explicitly (it is x1, the first regressor).
    rows = []
    for j, nm in enumerate(res["names"]):
        label = nm
        if j == 1:
            label = "ln_ntl"
        elif j == 2 and with_population:
            label = "ln_pop"
        rows.append({"variable": label,
                     "coef": round(float(res["params"][j]), 4),
                     "conley_se": round(float(res["conley_se"][j]), 4),
                     "classical_se": round(float(res["ols_se"][j]), 4),
                     "t_conley": round(float(res["t"][j]), 3)})
    table = pd.DataFrame(rows)
    table = table[table["variable"].isin(["ln_ntl", "ln_pop"])].reset_index(drop=True)
    log.info("\nConley spatial-HAC results (levels):\n" + table.to_string(index=False))
    return {"result": res, "table": table,
            "cutoff_km": cutoff_km, "lag_years": lag_years}
