
import logging
import pandas as pd

from pathlib import Path

from src.config import (
    PANEL_PKL, STATE_PANEL_PKL, OUTPUT_LOG,
    SHAPEFILE_PATH, STATE_SHP_PATH, USE_SYNTHETIC,
    OUTPUT_FIG, OUTPUT_TAB, DATA_RAW, DATA_CLEAN
)

# ── Create all output directories before anything else ───────
OUTPUT_LOG.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG.mkdir(parents=True, exist_ok=True)
OUTPUT_TAB.mkdir(parents=True, exist_ok=True)
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_CLEAN.mkdir(parents=True, exist_ok=True)
(DATA_RAW / "viirs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_LOG / "pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("main")


def run_pipeline(rebuild: bool = False):

    # ── Phase 1: Data ─────────────────────────────────────────
    log.info("=== PHASE 1: Data Acquisition ===")
    from src.data_acquisition import (
        download_county_shapefile, download_state_shapefile,
        fetch_bea_gdp, fetch_population
    )
    download_county_shapefile()
    download_state_shapefile()
    gdp_df = fetch_bea_gdp()
    pop_df = fetch_population()

    # ── Phase 2: NTL ─────────────────────────────────────────
    log.info("=== PHASE 2: NTL Processing ===")
    from src.ntl_processing import get_ntl_data
    ntl_df = get_ntl_data(gdp_df)

    # ── Phase 3: Panel ───────────────────────────────────────
    log.info("=== PHASE 3: Panel Construction ===")
    from src.panel_construction import (
        build_county_panel, build_state_panel, descriptive_statistics
    )
    if not rebuild and PANEL_PKL.exists():
        log.info("Loading cached county panel...")
        panel = pd.read_pickle(PANEL_PKL)
    else:
        panel = build_county_panel(gdp_df, ntl_df, pop_df)

    state_panel = build_state_panel(panel)
    stats = descriptive_statistics(panel)
    print("\n── Descriptive Statistics ──")
    print(stats.round(3).to_string())

    # ── Phase 4: OLS Benchmarks ──────────────────────────────
    log.info("=== PHASE 4: OLS Benchmarks ===")
    from src.benchmarks import run_pooled_ols
    ols_results = run_pooled_ols(panel)

    # ── Phase 5: Panel Fixed Effects ─────────────────────────
    log.info("=== PHASE 5: Panel Fixed Effects ===")
    from src.panel_fe import run_twoway_fe, run_heterogeneous_fe
    fe_results     = run_twoway_fe(panel)
    hetero_results = run_heterogeneous_fe(panel)

    # ── Phase 6: Spatial Model ───────────────────────────────
    log.info("=== PHASE 6: Spatial Model ===")
    spatial_result = None
    try:
        import geopandas as gpd
        from src.spatial_model import run_spatial_lag
        state_gdf      = gpd.read_file(STATE_SHP_PATH)
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
        plot_nowcast, plot_urban_rural_ntl, plot_county_map
    )
    plot_ntl_gdp_scatter(panel)
    plot_coefficient_comparison(ols_results, fe_results, hetero_results)
    plot_nowcast(nowcast_results)
    plot_urban_rural_ntl(panel)

    try:
        import geopandas as gpd
        from src.visualizations import plot_county_map
        county_gdf = gpd.read_file(SHAPEFILE_PATH)
        plot_county_map(panel, county_gdf)
    except Exception as e:
        log.warning(f"County map skipped: {e}")

    # ── Verify output files were created ─────────────────────────
    import os
    print("\n── Files created ──")
    for folder in [OUTPUT_FIG, OUTPUT_TAB, OUTPUT_LOG]:
        files = list(folder.glob("*"))
        print(f"{folder}:")
        for f in files:
            print(f"  {f.name} ({os.path.getsize(f)} bytes)")
        if not files:
            print("  (empty)")

    log.info("=== Pipeline complete. Outputs in /output/ ===")
    print("\n── Output Locations ──")
    print("  Figures: output/figures/")
    print("  Tables:  output/tables/")
    print("  Logs:    output/logs/")


if __name__ == "__main__":
    run_pipeline(rebuild=True)