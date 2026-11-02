"""
Run the two extension analyses off the cached county panel:

  1. Long-difference attenuation test (Griliches-Hausman). Estimates the light-GDP
     coefficient at horizons k = 1 to 6 and plots it against k. A coefficient that
     rises with the horizon points to measurement-error attenuation; a flat one
     points to a genuinely weak within-county relationship.

  2. The pandemic cross-sectional test. Regresses the one-year change in county
     log GDP on the change in log light for the 2020 downturn and the 2021
     rebound, with state fixed effects, and draws a binned scatter. It asks
     whether the counties that lost the most output also went the darkest.

Both read the cached panel built by main.py (data/clean/panel_county.pkl), so
main.py must have run once first. Each block is wrapped so one failure does not
stop the other.
"""

import logging

import pandas as pd

from src.config import PANEL_PKL, OUTPUT_TAB, OUTPUT_FIG, OUTPUT_LOG

for d in (OUTPUT_TAB, OUTPUT_FIG, OUTPUT_LOG):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(OUTPUT_LOG / "extensions.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("extensions")


def main():
    if not PANEL_PKL.exists():
        raise SystemExit("No cached panel at data/clean/panel_county.pkl. "
                         "Run main.py first to build the real panel.")
    panel = pd.read_pickle(PANEL_PKL)
    log.info(f"Loaded panel: {len(panel)} rows, {panel['fips'].nunique()} counties.")

    # 1. Long-difference attenuation test -------------------------------
    try:
        from src.long_differences import run_long_differences, plot_long_differences
        ld = run_long_differences(panel)
        if ld is not None and not ld.empty:
            plot_long_differences(ld)
    except Exception as e:
        log.warning(f"Long-difference test skipped: {e}")

    # 2. Pandemic cross-sectional test ----------------------------------
    try:
        from src.pandemic_test import run_pandemic_test, plot_pandemic_test
        pt = run_pandemic_test(panel)
        if pt["frames"]:
            plot_pandemic_test(pt)
    except Exception as e:
        log.warning(f"Pandemic test skipped: {e}")

    log.info("Extensions complete. New tables and figures are in output/.")


if __name__ == "__main__":
    main()
