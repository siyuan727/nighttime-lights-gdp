"""
Run the five additions in one place, off the already-built county panel.

This does not touch the main pipeline. It loads the cached county panel
(data/clean/panel_county.pkl, produced by main.py) and runs:

  1. Descriptive statistics table            -> output/tables/descriptive_stats*.csv
  2. Industry heterogeneity (CAGDP2)          -> output/tables/industry_*.csv, figure
  3. Conley spatial-HAC standard errors       -> output/tables/conley_levels.csv
  4. Horse-race inference (bootstrap + DM)     -> output/tables/horse_race_inference.csv
  5. Measurement-error IV (attenuation)        -> output/tables/measurement_error*.csv

The developing-economies panel is run separately with run_country_panel.py,
because it downloads national lights for ~150 countries and is the heavy step.

Each block is wrapped so one failure (a missing API key, a download that did not
run) does not stop the others. Blocks that need data you have not fetched yet
log what is missing and skip.
"""

import logging

import pandas as pd

from src.config import (PANEL_PKL, SHAPEFILE_PATH, OUTPUT_TAB, OUTPUT_FIG,
                        OUTPUT_LOG, CONLEY_CUTOFF_KM, CONLEY_LAG_YEARS,
                        HORSE_RACE_N_BOOT)

for d in (OUTPUT_TAB, OUTPUT_FIG, OUTPUT_LOG):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(OUTPUT_LOG / "improvements.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("improvements")


def main():
    if not PANEL_PKL.exists():
        raise SystemExit("No cached panel at data/clean/panel_county.pkl. "
                         "Run main.py first to build the real panel.")
    panel = pd.read_pickle(PANEL_PKL)
    log.info(f"Loaded panel: {len(panel)} rows, {panel['fips'].nunique()} counties.")

    # 1. Descriptive statistics -------------------------------------------
    try:
        from src.descriptives import build_descriptives
        build_descriptives(panel)
    except Exception as e:
        log.warning(f"Descriptives skipped: {e}")

    # 2. Industry heterogeneity (needs BEA CAGDP2) ------------------------
    try:
        from src.industry import run_industry_analysis
        run_industry_analysis(panel)
    except Exception as e:
        log.warning(f"Industry analysis skipped (needs BEA CAGDP2 pull): {e}")

    # 3. Conley spatial-HAC standard errors (needs the county shapefile) --
    try:
        from src.spatial_se import run_conley_levels
        out = run_conley_levels(panel, SHAPEFILE_PATH,
                                cutoff_km=CONLEY_CUTOFF_KM,
                                lag_years=CONLEY_LAG_YEARS)
        out["table"].to_csv(OUTPUT_TAB / "conley_levels.csv", index=False)
    except Exception as e:
        log.warning(f"Conley SE skipped: {e}")

    # 4. Horse-race inference --------------------------------------------
    try:
        from src.horse_race import run_horse_race_inference
        run_horse_race_inference(panel, n_boot=HORSE_RACE_N_BOOT)
    except Exception as e:
        log.warning(f"Horse-race inference skipped: {e}")

    # 5. Measurement-error IV (needs two half-year light measures) --------
    try:
        from src.measurement_error import (get_county_ntl_split,
                                            attach_split_measures,
                                            estimate_attenuation,
                                            estimate_attenuation_lag_fallback)
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

    log.info("Improvements run complete. New tables and figures in output/.")


if __name__ == "__main__":
    main()
