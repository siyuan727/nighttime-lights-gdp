"""
Productivity extension: does nighttime light proxy how PRODUCTIVE a place is,
or only how BIG it is?

Reads the cached county panel built by main.py (data/clean/panel_county.pkl),
fetches county employment from BEA (CAEMP25N) if not already cached, builds the
productivity panel, runs the full analysis, and writes tables + a figure to
output/. Safe to run after main.py.

    python run_productivity.py
"""

import logging

import pandas as pd

from src.config import PANEL_PKL, OUTPUT_TAB, OUTPUT_FIG, OUTPUT_LOG


def main():
    for d in (OUTPUT_TAB, OUTPUT_FIG, OUTPUT_LOG):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(OUTPUT_LOG / "productivity.log"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger("productivity_run")

    if not PANEL_PKL.exists():
        raise SystemExit("No cached panel at data/clean/panel_county.pkl. "
                         "Run main.py first to build the real panel.")
    panel = pd.read_pickle(PANEL_PKL)
    log.info(f"Loaded panel: {len(panel)} rows, {panel['fips'].nunique()} counties.")

    from src.productivity import build_productivity_panel, run_productivity, plot_productivity
    prod = build_productivity_panel(panel, save=True)
    result = run_productivity(prod, save=True)
    try:
        plot_productivity(result)
    except Exception as e:
        log.warning(f"Plot skipped: {e}")

    log.info("Productivity extension complete. See output/tables/productivity_*.csv "
             "and output/figures/.")


if __name__ == "__main__":
    main()
