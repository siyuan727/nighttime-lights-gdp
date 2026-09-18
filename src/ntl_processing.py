import numpy as np
import pandas as pd
import logging
from src.config import DATA_RAW, VIIRS_DIR, USE_SYNTHETIC, YEARS, START_YEAR, NTL_SOURCE

log = logging.getLogger(__name__)


def get_ntl_data(gdp_df):
    cache = DATA_RAW / "ntl_county.csv"

    if USE_SYNTHETIC:
        raise RuntimeError(
            "USE_SYNTHETIC is True. This project is real-data only. "
            "Set USE_SYNTHETIC = False in src/config.py."
        )

    if cache.exists():
        cached = pd.read_csv(cache, dtype={"fips": str})
        n = cached["fips"].nunique()
        mx = cached["ntl_mean"].max()
        if n < 3080 or mx > 300:
            raise RuntimeError(
                f"Cached NTL looks synthetic (counties={n}, max={mx:.1f}). "
                f"Delete {cache} and rerun to pull real Black Marble data."
            )
        log.info("Loading cached NTL data (real).")
        return cached

    from src.ntl_counties_bm import get_county_ntl
    return get_county_ntl()

def _synthetic_ntl(gdp_df):

    np.random.seed(42)

    # ── Parameters ────────────────────────────────────────────
    # rho reduced from 0.50 to 0.25: brings growth corr down to ~0.35-0.40
    # sigma_growth increased: adds realistic NTL measurement noise
    # sigma_level increased: adds more county-level dispersion in levels
    rho = 0.35
    sigma_growth = 0.055  # less noise — brings growth corr to ~0.30-0.40
    sigma_level = 0.55  # keep county dispersion high

    # ── Step 1: Compute county-level mean log GDP ─────────────
    # Larger/wealthier counties → higher baseline NTL
    county_mean_gdp = (
        gdp_df.groupby("fips")["gdp_thousands"]
        .mean()
        .apply(lambda x: np.log(max(x, 1)))
    )
    global_mean = county_mean_gdp.mean()
    global_std  = county_mean_gdp.std()
    if global_std < 1e-6:
        global_std = 1.0

    # ── Step 2: Compute GDP growth rates for each county ──────
    gdp_sorted = gdp_df.sort_values(["fips", "year"]).copy()
    gdp_sorted["ln_gdp"] = np.log(gdp_sorted["gdp_thousands"].clip(1))
    gdp_sorted["delta_ln_gdp"] = gdp_sorted.groupby("fips")["ln_gdp"].diff()

    # ── Step 3: Generate NTL for each county ──────────────────
    rows = []
    for fips, group in gdp_sorted.groupby("fips"):
        group = group.sort_values("year").reset_index(drop=True)

        # County baseline ln(NTL):
        # Scaled so that county at mean GDP has ln_ntl ~ 2.5 (NTL ~ 12)
        # County 2 std above mean has ln_ntl ~ 4.0  (NTL ~ 55)
        # County 2 std below mean has ln_ntl ~ 1.0  (NTL ~ 2.7)
        mean_ln_gdp = county_mean_gdp.get(fips, global_mean)
        gdp_z = (mean_ln_gdp - global_mean) / global_std

        # Get population z-score for this county
        # Urban counties (high pop) get meaningfully higher NTL baseline
        county_pop = gdp_df[gdp_df["fips"] == fips]["gdp_thousands"].mean()
        pop_boost = np.clip(gdp_z * 0.5, -1.0, 2.0)  # urban premium

        ln_ntl_base = (
                3.0
                + gdp_z * 0.45
                + pop_boost * 0.30
                + np.random.normal(0, sigma_level)
        )

        # Walk forward: each year's NTL grows with GDP growth + noise
        ln_ntl_current = ln_ntl_base
        for idx, row in group.iterrows():
            delta_gdp = row["delta_ln_gdp"]
            if pd.isna(delta_gdp):
                delta_gdp = 0.0

            # Δln_ntl = rho * Δln_gdp + noise
            # This embeds GDP growth signal into NTL changes
            delta_ntl      = rho * delta_gdp + np.random.normal(0, sigma_growth)
            ln_ntl_current = ln_ntl_current + delta_ntl

            ntl = float(np.clip(np.exp(ln_ntl_current), 0.3, 2000.0))
            rows.append({
                "fips":     fips,
                "year":     int(row["year"]),
                "ntl_mean": round(ntl, 4),
                "ntl_sum":  round(ntl * np.random.uniform(50, 5000), 2),
            })

    df = pd.DataFrame(rows)

    # Verify it worked
    check = gdp_df.merge(df, on=["fips","year"], how="inner")
    check["ln_gdp"] = np.log(check["gdp_thousands"].clip(1))
    check["ln_ntl"] = np.log(check["ntl_mean"].clip(0.01))
    level_corr = check[["ln_gdp","ln_ntl"]].corr().iloc[0,1]

    check = check.sort_values(["fips","year"])
    check["d_gdp"] = check.groupby("fips")["ln_gdp"].diff()
    check["d_ntl"] = check.groupby("fips")["ln_ntl"].diff()
    growth_corr = check[["d_gdp","d_ntl"]].dropna().corr().iloc[0,1]

    log.info(f"Synthetic NTL — levels corr: {level_corr:.3f}, "
             f"growth corr: {growth_corr:.3f}")
    log.info(f"NTL range: {df['ntl_mean'].min():.2f} "
             f"to {df['ntl_mean'].max():.2f}, "
             f"mean: {df['ntl_mean'].mean():.2f}")
    return df


def _real_viirs():
    import geopandas as gpd
    from rasterstats import zonal_stats
    import rasterio
    from src.config import SHAPEFILE_PATH, EXCLUDE_FIPS_PREFIX

    gdf = gpd.read_file(SHAPEFILE_PATH)
    gdf = gdf[~gdf["STATEFP"].isin(EXCLUDE_FIPS_PREFIX)]
    gdf["fips"] = gdf["STATEFP"] + gdf["COUNTYFP"]
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    rows = []
    for year in YEARS:
        tif = VIIRS_DIR / f"viirs_{year}.tif"
        if not tif.exists():
            log.warning(f"Missing: {tif}")
            continue
        log.info(f"Processing VIIRS {year}...")
        with rasterio.open(tif) as src:
            nodata = src.nodata if src.nodata is not None else -9999
            gdf_proj = gdf.to_crs(src.crs.to_epsg()) if src.crs else gdf
        stats = zonal_stats(gdf_proj, str(tif),
                            stats=["mean","sum"], nodata=nodata)
        for i, (_, county) in enumerate(gdf_proj.iterrows()):
            m = stats[i].get("mean")
            s = stats[i].get("sum")
            if m is None:
                continue
            rows.append({"fips": county["fips"], "year": year,
                         "ntl_mean": round(float(m), 4),
                         "ntl_sum":  round(float(s), 2) if s else 0.0})
    return pd.DataFrame(rows)