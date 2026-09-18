"""
Draw the two-panel US figure (real VIIRS lights + real BEA county GDP) from the
already-built real panel. Run main.py first so the real panel exists.

    python run_us_maps.py
"""
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
import pandas as pd
from src.config import PANEL_PKL
from src.us_maps import plot_us_lights_and_gdp

if __name__ == "__main__":
    panel = pd.read_pickle(PANEL_PKL)
    plot_us_lights_and_gdp(panel)
    print("\nDone. See output/figures/us_lights_and_gdp.png")
