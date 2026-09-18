"""
US nighttime-lights / GDP pipeline.

Runs the whole thing end to end and writes figures and tables to output/.
With USE_SYNTHETIC = False in src/config.py, every phase uses real data:
BEA county GDP, Census population, and NASA Black Marble nighttime lights.
"""

import os
import logging

import pandas as pd

from src.config import (
    PANEL_PKL, STATE_PANEL_PKL, OUTPUT_LOG,
    SHAPEFILE_PATH, STATE_SHP_PATH, USE_SYNTHETIC,
    OUTPUT_FIG, OUTPUT_TAB, DATA_RAW, DATA_CLEAN, NTL_SOURCE,
)

# ── Create output directories before anything else ───────────
for d in (OUTPUT_LOG, OUTPUT_FIG, OUTPUT_TAB, DATA_RAW, DATA_CLEAN,
          DATA_RAW / "viirs"):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(OUTPUT_LOG / "pipeline.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("main")


def run_pipeline(rebuild: bool = True):
    # ── Data-mode banner: know immediately what you are running ──
    if USE_SYNTHETIC:
        log.warning("=" * 60)
        log.warning("USE_SYNTHETIC = True  ->  running on SYNTHETIC data.")
        log.warning("Set USE_SYNTHETIC = False in src/config.py for real data.")
        log.warning("=" * 60)
    else:
        log.info(f"Real-data mode. Nighttime-lights source: {NTL_SOURCE}.")
        if NTL_SOURCE == "blackmarble" and not os.environ.get("BLACKMARBLE_TOKEN"):
            log.warning("BLACKMARBLE_TOKEN is not set. The nighttime-lights "
                        "download will fail until you set it.")

    # ── Phase 1: Data acquisition ────────────────────────────
    log.info("=== PHASE 1: Data Acquisition ===")
    from src.data_acquisition import (
        download_county_shapefile, download_state_shapefile,
        fetch_bea_gdp, fetch_population,
    )
    download_county_shapefile()
    download_state_shapefile()
    gdp_df = fetch_bea_gdp()
    pop_df = fetch_population()

    # ── Phase 2: Nighttime lights ────────────────────────────
    log.info("=== PHASE 2: NTL Processing ===")
    from src.ntl_processing import get_ntl_data
    ntl_df = get_ntl_data(gdp_df)

    # ── Phase 3: Panel construction ──────────────────────────
    log.info("=== PHASE 3: Panel Construction ===")
    from src.panel_construction import (
        build_county_panel, build_state_panel, descriptive_statistics,
    )
    if not rebuild and PANEL_PKL.exists():
        log.info("Loading cached county panel...")
        panel = pd.read_pickle(PANEL_PKL)
    else:
        panel = build_county_panel(gdp_df, ntl_df, pop_df)

    state_panel = build_state_panel(panel)
    stats = descriptive_statistics(panel)
    print("\n-- Descriptive Statistics --")
    print(stats.round(3).to_string())

    # ── Phase 4: OLS benchmarks ──────────────────────────────
    log.info("=== PHASE 4: OLS Benchmarks ===")
    from src.benchmarks import run_pooled_ols
    ols_results = run_pooled_ols(panel)

    # ── Phase 5: Panel fixed effects ─────────────────────────
    log.info("=== PHASE 5: Panel Fixed Effects ===")
    from src.panel_fe import run_twoway_fe, run_heterogeneous_fe
    fe_results = run_twoway_fe(panel)
    hetero_results = run_heterogeneous_fe(panel)

    # ── Phase 6: Spatial model ───────────────────────────────
    log.info("=== PHASE 6: Spatial Model ===")
    try:
        import geopandas as gpd
        from src.spatial_model import run_spatial_lag
        state_gdf = gpd.read_file(STATE_SHP_PATH)
        spatial_result, merged_gdf = run_spatial_lag(state_panel, state_gdf)
    except Exception as e:
        log.warning(f"Spatial model skipped: {e}")

    # ── Phase 7: Nowcasting ──────────────────────────────────
    log.info("=== PHASE 7: Nowcasting ===")
    from src.nowcasting import run_nowcasting
    nowcast_results = run_nowcasting(panel)

    # ── Phase 8: Visualizations ──────────────────────────────
    log.info("=== PHASE 8: Visualizations ===")
    from src.visualizations import (
        plot_ntl_gdp_scatter, plot_coefficient_comparison,
        plot_nowcast, plot_urban_rural_ntl,
    )
    plot_ntl_gdp_scatter(panel)
    plot_coefficient_comparison(ols_results, fe_results, hetero_results)
    plot_nowcast(nowcast_results)
    plot_urban_rural_ntl(panel)

    # Real two-panel US figure: VIIRS lights raster + real county GDP.
    # Replaces the old synthetic three-panel county map.
    try:
        from src.us_maps import plot_us_lights_and_gdp
        plot_us_lights_and_gdp(panel)
    except Exception as e:
        log.warning(f"US lights/GDP figure skipped: {e}")

    # ── Phase 9: US GDP level estimation (optional) ──────────
    log.info("=== PHASE 9: US GDP Level Estimation ===")
    try:
        from src.gdp_estimation import estimate_us_gdp, plot_us_gdp_estimation
        gdp_est = estimate_us_gdp(panel, synthetic_covid=False)
        if gdp_est:
            plot_us_gdp_estimation(gdp_est)
    except Exception as e:
        log.warning(f"US GDP estimation skipped: {e}")

    # ── Verify outputs ───────────────────────────────────────
    print("\n-- Files created --")
    for folder in (OUTPUT_FIG, OUTPUT_TAB, OUTPUT_LOG):
        files = list(folder.glob("*"))
        print(f"{folder}:")
        for f in files:
            print(f"  {f.name} ({os.path.getsize(f)} bytes)")
        if not files:
            print("  (empty)")

    log.info("=== Pipeline complete. Outputs in /output/ ===")
    print("\n-- Output locations --")
    print("  Figures: output/figures/")
    print("  Tables:  output/tables/")
    print("  Logs:    output/logs/")


if __name__ == "__main__":
    run_pipeline(rebuild=True)
