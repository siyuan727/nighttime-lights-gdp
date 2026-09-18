"""
Two-panel US figure using only real data: the actual VIIRS nighttime-lights
raster on the left, and real BEA county GDP as a choropleth on the right.
Contiguous US only. Both panels come from the same real pipeline, so there is
no synthetic-vs-real mismatch.
"""

import logging

import numpy as np
import geopandas as gpd

from src.config import OUTPUT_FIG, EXCLUDE_FIPS_PREFIX, MAP_YEAR
from src import data_acquisition

log = logging.getLogger(__name__)


def _county_geo():
    shp = data_acquisition.download_county_shapefile()
    gdf = gpd.read_file(shp)
    gdf = gdf[~gdf["STATEFP"].isin(EXCLUDE_FIPS_PREFIX)].copy()
    gdf["fips"] = (gdf["STATEFP"].astype(str).str.zfill(2)
                   + gdf["COUNTYFP"].astype(str).str.zfill(3))
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    return gdf[["fips", "geometry"]]


def plot_us_lights_and_gdp(panel, year=None, save_path=None):
    import matplotlib.pyplot as plt

    year = year or MAP_YEAR
    geo = _county_geo()
    snap = panel[panel["year"] == year][["fips", "gdp_thousands"]].copy()
    snap["fips"] = snap["fips"].astype(str).str.zfill(5)
    gdf = geo.merge(snap, on="fips", how="left")

    # Try the real VIIRS raster for the left panel.
    da = None
    try:
        from src.ntl_counties_bm import get_us_raster
        ras = get_us_raster(year)
        da = ras.to_array().squeeze() if hasattr(ras, "to_array") else ras.squeeze()
    except Exception as e:
        log.warning(f"US lights raster unavailable, drawing GDP map only: {e}")

    if da is not None:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 6))
        im = axL.imshow(da.values, cmap="inferno",
                        extent=[float(da.x.min()), float(da.x.max()),
                                float(da.y.min()), float(da.y.max())],
                        origin="upper", aspect="equal",
                        vmin=float(da.quantile(0.02)),
                        vmax=float(da.quantile(0.98)))
        fig.colorbar(im, ax=axL, shrink=0.5,
                     label="radiance (nW/cm2/sr)")
        axL.set_title(f"Nighttime lights (VIIRS VNP46A4), {year}", fontsize=11)
        axL.set_xlim(-125, -66)
        axL.set_ylim(24, 50)
        axL.axis("off")
    else:
        fig, axR = plt.subplots(figsize=(10, 5))

    # Base layer and choropleth with no county borders (seamless look).
    gdf.plot(ax=axR, color="#EEEEEE", edgecolor="none", linewidth=0)
    sub = gdf[gdf["gdp_thousands"].notna() & (gdf["gdp_thousands"] > 0)].copy()
    sub["log_gdp"] = np.log10(sub["gdp_thousands"])
    sub.plot(ax=axR, column="log_gdp", cmap="viridis", legend=True,
             edgecolor="none", linewidth=0,
             legend_kwds={"label": "log10(real GDP, thousands $)", "shrink": 0.5})
    axR.set_title(f"Real county GDP (BEA CAGDP1), {year}", fontsize=11)
    axR.set_xlim(-125, -66)
    axR.set_ylim(24, 50)
    axR.set_aspect("equal", adjustable="box")       # match left panel
    axR.axis("off")

    fig.suptitle("Nighttime lights and economic output, contiguous US "
                 "(real data: NASA Black Marble + BEA)", fontsize=13)
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / "us_lights_and_gdp.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out
