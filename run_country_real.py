"""
Run the real-data country GDP-from-lights analysis for Ghana.

Before running:
  1. pip install wbgapi blackmarblepy geopandas
  2. Create a NASA Earthdata account (https://urs.earthdata.nasa.gov),
     Generate Token, and set it:
        Windows PowerShell:  $env:BLACKMARBLE_TOKEN="your_token_here"
        macOS/Linux:         export BLACKMARBLE_TOKEN="your_token_here"
  3. python run_country_real.py
n
The first run downloads World Bank GDP, NASA Black Marble lights, and Natural
Earth boundaries, then caches them. Later runs are fast. No data is simulated.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

from src import country_real

if __name__ == "__main__":
    results = country_real.run()
    country_real.plot_estimate(results)          # Ghana estimate vs actual GDP
    country_real.plot_lights_and_gdp(results)    # Ghana lights + GDP side by side
    print("\nDone. See output/figures/ and output/tables/.")
