"""
Real nighttime lights from NASA's Black Marble (VIIRS VNP46A4 annual
composite), through the World Bank's blackmarblepy package.

VNP46A4 is the annual, gap-filled, straylight-corrected VIIRS product, which
is the right one for year-over-year economic work. bm_extract aggregates the
radiance inside each country polygon; bm_raster returns the gridded image for
a country so we can draw the actual lights map.

A free NASA Earthdata bearer token is required. Create an account at
https://urs.earthdata.nasa.gov, then Generate Token, and put the string in the
BLACKMARBLE_TOKEN environment variable. Nothing here is simulated.
"""

import os
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, COUNTRY_YEARS, NASA_TOKEN_ENV, TARGET_ISO3

log = logging.getLogger(__name__)

NTL_CACHE = DATA_RAW / "ntl_country_blackmarble.csv"
BM_DIR = DATA_RAW / "blackmarble"


def _token():
    tok = os.environ.get(NASA_TOKEN_ENV, "").strip()
    if not tok:
        raise RuntimeError(
            f"No NASA Earthdata token found. Set it with the {NASA_TOKEN_ENV} "
            "environment variable. Get one free at "
            "https://urs.earthdata.nasa.gov (Generate Token)."
        )
    return tok


def _normalize(res, gdf):
    """
    Turn a bm_extract result into a tidy [iso3, year, ntl_mean, ntl_sum] frame.

    Column names in the raw result depend on the package version and the
    aggregation, so this resolves them defensively and logs what it found. If
    the auto-detection is ever wrong, the log line shows the real columns so
    the mapping can be fixed in one place.
    """
    df = pd.DataFrame(res).copy()
    log.info(f"bm_extract returned columns: {list(df.columns)}")

    # Year / date column.
    year_col = next((c for c in df.columns if c.lower() in ("date", "time", "year")), None)
    if year_col is None:
        raise ValueError("No date/year column in bm_extract output.")
    df["year"] = pd.to_datetime(df[year_col].astype(str), errors="coerce").dt.year
    df["year"] = df["year"].fillna(
        df[year_col].astype(str).str.extract(r"(\d{4})")[0].astype(float))
    df["year"] = df["year"].astype(int)

    # iso3 column carried from the input polygons.
    iso_col = next((c for c in df.columns if c.lower() == "iso3"), None)
    if iso_col is None:
        raise ValueError("bm_extract output has no iso3 column; ensure the "
                         "input GeoDataFrame carries iso3.")

    # Aggregated radiance columns: prefer explicit sum / mean names.
    def pick(kind):
        for c in df.columns:
            if kind in c.lower() and c not in (iso_col, year_col, "year"):
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
        return None

    sum_c, mean_c = pick("sum"), pick("mean")
    if sum_c is None and mean_c is None:
        # Fall back to the first numeric non-id column as the value.
        numeric = [c for c in df.columns
                   if pd.api.types.is_numeric_dtype(df[c])
                   and c not in (iso_col, year_col, "year")]
        if not numeric:
            raise ValueError("No numeric radiance column found in output.")
        sum_c = numeric[0]

    out = pd.DataFrame({"iso3": df[iso_col].astype(str).str.upper(),
                        "year": df["year"]})
    out["ntl_sum"] = df[sum_c] if sum_c else df[mean_c]
    out["ntl_mean"] = df[mean_c] if mean_c else df[sum_c]
    out = out.dropna(subset=["ntl_sum"]).sort_values(["iso3", "year"])
    return out.reset_index(drop=True)


def get_country_ntl(gdf, years=None, use_cache=True):
    """
    Country-year nighttime lights for the polygons in gdf. gdf must have an
    iso3 column. Cached to CSV so the download happens once.
    """
    years = list(years or COUNTRY_YEARS)
    if use_cache and NTL_CACHE.exists():
        log.info(f"Loading cached NTL from {NTL_CACHE.name}")
        return pd.read_csv(NTL_CACHE, dtype={"iso3": str})

    from blackmarble.extract import bm_extract
    from blackmarble.types import Product

    if "iso3" not in gdf.columns:
        raise ValueError("gdf must carry an iso3 column for aggregation.")

    # Black Marble maps the geometry's bounding box to VIIRS tiles, so it must
    # be in plain lat/lon. Set or convert the CRS to WGS84 before sending.
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    BM_DIR.mkdir(parents=True, exist_ok=True)
    dates = [date(y, 1, 1) for y in years]
    token = _token()

    # Extract one country at a time. Passing all countries at once makes Black
    # Marble compute tiles for the combined bounding box, which spans the globe
    # and pulls in nonexistent polar tiles, aborting the whole run. Per country
    # the bounding box is small and correct, and one failure does not kill the
    # rest.
    frames = []

    import time
    # Do the target country first so its success or failure is known up front.
    isos = sorted(gdf["iso3"].unique())
    if TARGET_ISO3 in isos:
        isos = [TARGET_ISO3] + [i for i in isos if i != TARGET_ISO3]

    for n, iso in enumerate(isos, 1):
        sub = gdf[gdf["iso3"] == iso]
        for attempt in (1, 2):
            try:
                log.info(f"Black Marble VNP46A4: {iso} ({n}/{len(isos)}) "
                         f"attempt {attempt}...")
                res = bm_extract(
                    sub, Product.VNP46A4, dates, token,
                    aggfunc=["mean", "sum"], output_directory=BM_DIR,
                    check_all_tiles_exist=False,
                )
                frames.append(_normalize(res, sub))
                break
            except Exception as e:
                log.warning(f"  {iso} attempt {attempt} failed: {e}")
                if attempt == 2:
                    log.warning(f"  {iso} skipped.")
                else:
                    time.sleep(5)

    if not frames:
        raise RuntimeError("Black Marble returned no data for any country.")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(NTL_CACHE, index=False)
    log.info(f"Cached NTL to {NTL_CACHE.name}: {len(out)} rows, "
             f"{out['iso3'].nunique()} countries")
    return out


def get_country_raster(gdf_one_country, year, out_tif=None):
    """
    Gridded VIIRS radiance for a single country and year, as an xarray object,
    for drawing the lights map. gdf_one_country should hold one polygon.
    """
    from blackmarble.raster import bm_raster
    from blackmarble.types import Product

    if gdf_one_country.crs is None:
        gdf_one_country = gdf_one_country.set_crs(4326)
    else:
        gdf_one_country = gdf_one_country.to_crs(4326)

    BM_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Black Marble raster for {year}...")
    ras = bm_raster(
        gdf_one_country, Product.VNP46A4, [date(year, 1, 1)], _token(),
        output_directory=BM_DIR, check_all_tiles_exist=False,
    )
    return ras
