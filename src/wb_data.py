"""
Real data loaders for the country module.

  - GDP comes from the World Bank World Development Indicators through the
    official wbgapi package. Indicator NY.GDP.MKTP.KD is GDP in constant 2015
    US dollars, so it is real output and comparable across years.
  - Country boundaries come from Natural Earth (admin 0, 1:110m), downloaded
    once from the maintained natural-earth-vector repository and cached.

Nothing here is simulated. If a source is unreachable the functions raise
rather than fall back to anything fabricated.
"""

import logging
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
import wbgapi as wb

from src.config import DATA_RAW, COUNTRY_YEARS, GDP_INDICATOR

log = logging.getLogger(__name__)

NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
          "master/geojson/ne_110m_admin_0_countries.geojson")


def get_country_metadata():
    """
    ISO3 code, name, region, and income group for every economy (aggregates
    like 'World' excluded). Used to pick the data-rich estimation panel.
    """
    meta = wb.economy.DataFrame(labels=True, skipAggs=True).reset_index()
    # wbgapi returns the ISO3 code as the index; name/region/incomeLevel as cols.
    rename = {}
    for c in meta.columns:
        lc = c.lower()
        if lc in ("index", "id", "economy"):
            rename[c] = "iso3"
        elif lc == "name":
            rename[c] = "name"
        elif lc == "region":
            rename[c] = "region"
        elif lc in ("incomelevel", "income_level"):
            rename[c] = "income"
    meta = meta.rename(columns=rename)
    keep = [c for c in ["iso3", "name", "region", "income"] if c in meta.columns]
    out = meta[keep].copy()
    out["iso3"] = out["iso3"].astype(str).str.upper()
    log.info(f"World Bank metadata: {len(out)} economies")
    return out


def get_wdi_gdp(iso3_list, years=None):
    """
    Long GDP panel [iso3, year, gdp] in constant 2015 USD for the requested
    countries and years. gdp is in current dataset units (USD); the analysis
    only uses logs and growth, so the absolute scale does not matter.
    """
    years = list(years or COUNTRY_YEARS)
    raw = wb.data.DataFrame(
        GDP_INDICATOR, economy=iso3_list, time=years,
        skipAggs=True, labels=False,
    )
    # wbgapi returns economies as rows and time periods as columns ('YR2013').
    df = raw.reset_index()
    id_col = df.columns[0]
    long = df.melt(id_vars=id_col, var_name="period", value_name="gdp")
    long = long.rename(columns={id_col: "iso3"})
    long["year"] = long["period"].astype(str).str.extract(r"(\d{4})").astype(float)
    long = long.dropna(subset=["gdp", "year"])
    long["year"] = long["year"].astype(int)
    long["iso3"] = long["iso3"].astype(str).str.upper()
    long = long[["iso3", "year", "gdp"]].sort_values(["iso3", "year"])
    log.info(f"World Bank GDP: {long['iso3'].nunique()} countries, "
             f"{long['year'].min()}-{long['year'].max()}, {len(long)} rows")
    return long.reset_index(drop=True)


def get_country_geometries(cache_path=None):
    """Natural Earth admin-0 country polygons with a clean iso3 column."""
    cache = Path(cache_path) if cache_path else (
        DATA_RAW / "ne_110m_admin_0_countries.geojson")
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.exists():
        log.info("Downloading Natural Earth country boundaries...")
        r = requests.get(NE_URL, timeout=180)
        r.raise_for_status()
        cache.write_bytes(r.content)

    gdf = gpd.read_file(cache)
    # ISO_A3 is '-99' for a few countries; ISO_A3_EH fixes most of them.
    iso = gdf.get("ISO_A3").astype(str)
    if "ISO_A3_EH" in gdf.columns:
        bad = iso.isin(["-99", "nan", "None"]) | (iso.str.len() != 3)
        iso = iso.where(~bad, gdf["ISO_A3_EH"].astype(str))
    gdf = gdf.assign(iso3=iso.str.upper())
    name_col = "ADMIN" if "ADMIN" in gdf.columns else "NAME"
    cont_col = "CONTINENT" if "CONTINENT" in gdf.columns else None
    cols = {name_col: "name"}
    if cont_col:
        cols[cont_col] = "continent"
    gdf = gdf.rename(columns=cols)
    keep = ["iso3", "name"] + (["continent"] if cont_col else []) + ["geometry"]
    gdf = gdf[keep]
    gdf = gdf[gdf["iso3"].str.len() == 3].reset_index(drop=True)
    log.info(f"Natural Earth: {len(gdf)} country polygons")
    return gdf
