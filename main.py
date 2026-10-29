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
    from src.panel_fe import run_twoway_fe, run_heterogeneous_fe, run_lag_fe
    fe_results = run_twoway_fe(panel)
    hetero_results = run_heterogeneous_fe(panel)
    fe_results.update(run_lag_fe(panel))

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

    # ── Phase 7b: Light-vs-population horse race ─────────────
    log.info("=== PHASE 7b: Horse Race (light beyond population) ===")
    from src.horse_race import run_horse_race, plot_horse_race
    hr = run_horse_race(panel)
    plot_horse_race(hr)

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

    # ── Phase 10: Descriptive statistics (publication table) ──
    log.info("=== PHASE 10: Descriptive Statistics ===")
    try:
        from src.descriptives import build_descriptives
        build_descriptives(panel)
    except Exception as e:
        log.warning(f"Descriptives skipped: {e}")

    # ── Phase 11: Industry heterogeneity (BEA CAGDP2) ─────────
    log.info("=== PHASE 11: Industry Heterogeneity ===")
    try:
        from src.industry import run_industry_analysis
        run_industry_analysis(panel)
    except Exception as e:
        log.warning(f"Industry analysis skipped (needs BEA CAGDP2): {e}")

    # ── Phase 12: Conley spatial-HAC standard errors ──────────
    log.info("=== PHASE 12: Conley Spatial Standard Errors ===")
    try:
        from src.spatial_se import run_conley_levels
        conley = run_conley_levels(panel, SHAPEFILE_PATH)
        conley["table"].to_csv(OUTPUT_TAB / "conley_levels.csv", index=False)
    except Exception as e:
        log.warning(f"Conley SE skipped: {e}")

    # ── Phase 13: Horse-race inference (bootstrap + Diebold-Mariano) ──
    log.info("=== PHASE 13: Horse-Race Inference ===")
    try:
        from src.horse_race import run_horse_race_inference
        run_horse_race_inference(panel)
    except Exception as e:
        log.warning(f"Horse-race inference skipped: {e}")

    # ── Phase 14: Measurement-error IV (attenuation vs composition) ──
    log.info("=== PHASE 14: Measurement-Error IV ===")
    try:
        from src.measurement_error import (
            get_county_ntl_split, attach_split_measures,
            estimate_attenuation, estimate_attenuation_lag_fallback,
        )
        try:
            split = get_county_ntl_split()          # monthly Black Marble
            panel_me = attach_split_measures(panel, split)
            estimate_attenuation(panel_me, beta_pooled=0.845)
        except Exception as e_split:
            log.warning(f"Half-year measures unavailable ({e_split}); "
                        f"using the lagged-light instrument fallback.")
            estimate_attenuation_lag_fallback(panel)
    except Exception as e:
        log.warning(f"Measurement-error IV skipped: {e}")

    # ── Phase 15: Developing-economies external validity (HEAVY) ──
    # Downloads national Black Marble lights for ~150 countries, so it is left
    # commented to keep a normal `python main.py` fast. Uncomment to include it,
    # or just run `python run_country_panel.py` on its own. Requires the config
    # additions (REFERENCE_INCOME_GROUPS etc.) pasted into src/config.py and
    # BLACKMARBLE_TOKEN set.
    # log.info("=== PHASE 15: Developing-Economies Panel ===")
    # try:
    #     from src.country_panel import run as run_country_panel
    #     run_country_panel(save=True)
    # except Exception as e:
    #     log.warning(f"Developing-economies panel skipped: {e}")

    # ── Verify outputs ───────────────────────────────────────
    print("\n-- Files created --")
    for folder in (OUTPUT_FIG, OUTPUT_TAB, OUTPUT_LOG):
        files = list(folder.glob("*"))mkk
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
