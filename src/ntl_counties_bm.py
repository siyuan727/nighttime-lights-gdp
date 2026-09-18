"""
Real county-level nighttime lights for the contiguous US, from NASA Black
Marble (VIIRS VNP46A4 annual), aggregated to county polygons with the same
blackmarblepy tool used for the country module.

Passing all contiguous-US counties at once makes Black Marble download the
tiles covering the US bounding box once per year (about fifteen tiles), then
average the radiance inside each county. That is far cheaper than looping
county by county, which would re-download overlapping tiles thousands of times.

Requires the NASA Earthdata token in the BLACKMARBLE_TOKEN environment
variable, plus LAADS authorization and the product EULA accepted on your
Earthdata account (the same setup the Ghana run needed). Nothing is simulated.
"""

import os
import logging
from datetime import date

import pandas as pd
import geopandas as gpd

from src.config import DATA_RAW, YEARS, EXCLUDE_FIPS_PREFIX, NASA_TOKEN_ENV
from src import data_acquisition

log = logging.getLogger(__name__)

NTL_CACHE = DATA_RAW / "ntl_county.csv"          # schema get_ntl_data expects
BM_DIR = DATA_RAW / "blackmarble_us"


def _token():
    tok = os.environ.get(NASA_TOKEN_ENV, "").strip()
    if not tok:
        raise RuntimeError(
            f"No NASA Earthdata token found. Set {NASA_TOKEN_ENV} to your "
            "Earthdata token (https://urs.earthdata.nasa.gov, Generate Token)."
        )
    return tok


def _county_gdf():
    """Contiguous-US county polygons in WGS84 with a 5-digit fips column."""
    shp = data_acquisition.download_county_shapefile()
    gdf = gpd.read_file(shp)
    gdf = gdf[~gdf["STATEFP"].isin(EXCLUDE_FIPS_PREFIX)].copy()
    gdf["fips"] = (gdf["STATEFP"].astype(str).str.zfill(2)
                   + gdf["COUNTYFP"].astype(str).str.zfill(3))
    gdf = gdf[["fips", "geometry"]]
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    return gdf.reset_index(drop=True)


def _normalize(res):
    """bm_extract result -> [fips, year, ntl_mean, ntl_sum]."""
    df = pd.DataFrame(res).copy()
    log.info(f"bm_extract columns: {list(df.columns)}")

    date_col = next((c for c in df.columns if c.lower() in ("date", "time", "year")), None)
    df["year"] = pd.to_datetime(df[date_col].astype(str), errors="coerce").dt.year
    df["year"] = df["year"].fillna(
        df[date_col].astype(str).str.extract(r"(\d{4})")[0].astype(float)).astype(int)

    fips_col = next((c for c in df.columns if c.lower() == "fips"), None)
    if fips_col is None:
        raise ValueError("bm_extract output missing fips; ensure the county "
                         "GeoDataFrame carried a fips column.")

    def pick(kind):
        for c in df.columns:
            if kind in c.lower() and c not in (fips_col, date_col, "year") \
                    and pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    sum_c, mean_c = pick("sum"), pick("mean")
    if mean_c is None and sum_c is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                   and c not in (fips_col, date_col, "year")]
        mean_c = numeric[0] if numeric else None
    out = pd.DataFrame({
        "fips": df[fips_col].astype(str).str.zfill(5),
        "year": df["year"],
        "ntl_mean": df[mean_c] if mean_c else df[sum_c],
        "ntl_sum": df[sum_c] if sum_c else df[mean_c],
    })
    return out.dropna(subset=["ntl_mean"]).reset_index(drop=True)


def get_county_ntl(years=None, use_cache=True):
    years = list(years or YEARS)
    if use_cache and NTL_CACHE.exists():
        log.info(f"Loading cached county NTL from {NTL_CACHE.name}")
        return pd.read_csv(NTL_CACHE, dtype={"fips": str})

    from blackmarble.extract import bm_extract
    from blackmarble.types import Product

    gdf = _county_gdf()
    BM_DIR.mkdir(parents=True, exist_ok=True)
    token = _token()
    log.info(f"Black Marble VNP46A4: {len(gdf)} contiguous-US counties, "
             f"{len(years)} years. Downloads US tiles once per year.")

    frames = []
    for y in years:
        for attempt in (1, 2):
            try:
                log.info(f"  {y} (attempt {attempt})...")
                res = bm_extract(
                    gdf, Product.VNP46A4, [date(y, 1, 1)], token,
                    aggfunc=["mean", "sum"], output_directory=BM_DIR,
                    check_all_tiles_exist=False,
                )
                frames.append(_normalize(res))
                break
            except Exception as e:
                log.warning(f"  {y} attempt {attempt} failed: {e}")
                if attempt == 2:
                    log.warning(f"  {y} skipped.")

    if not frames:
        raise RuntimeError("Black Marble returned no county data for any year.")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(NTL_CACHE, index=False)
    log.info(f"Cached county NTL: {len(out)} rows, {out['fips'].nunique()} "
             f"counties, years {sorted(out['year'].unique())}")
    return out


def get_us_raster(year):
    """VIIRS radiance raster for the whole contiguous US, for the lights map."""
    from blackmarble.raster import bm_raster
    from blackmarble.types import Product

    us = _county_gdf().dissolve().reset_index(drop=True)   # one US boundary
    BM_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Black Marble raster: contiguous US, {year}...")
    return bm_raster(us, Product.VNP46A4, [date(year, 1, 1)], _token(),
                     output_directory=BM_DIR, check_all_tiles_exist=False)
