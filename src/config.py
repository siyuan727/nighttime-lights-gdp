from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT_DIR / "data" / "raw"
DATA_CLEAN  = ROOT_DIR / "data" / "clean"
OUTPUT_FIG  = ROOT_DIR / "output" / "figures"
OUTPUT_TAB  = ROOT_DIR / "output" / "tables"
OUTPUT_LOG  = ROOT_DIR / "output" / "logs"

PANEL_PKL       = DATA_CLEAN / "panel_county.pkl"
STATE_PANEL_PKL = DATA_CLEAN / "panel_state.pkl"
SHAPEFILE_PATH  = DATA_RAW  / "counties" / "tl_2022_us_county.shp"
STATE_SHP_PATH  = DATA_RAW  / "states"   / "tl_2022_us_state.shp"

BEA_API_KEY = "018F8C8F-C213-48EF-9564-C959ACC6E010"
NTL_SOURCE = "blackmarble"

START_YEAR = 2016
END_YEAR   = 2022
YEARS      = list(range(START_YEAR, END_YEAR + 1))

EXCLUDE_FIPS_PREFIX = ["02", "15", "60", "66", "69", "72", "78"]

TRAIN_END_YEAR   = 2018
TEST_START_YEAR  = 2019
MIN_YEARS_REQUIRED = 7

# True = synthetic data (works immediately, no downloads needed)
# False = real VIIRS .tif files (see Phase 6 for download instructions)
USE_SYNTHETIC = False

VIIRS_DIR = DATA_RAW / "viirs"

PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 150
NAVY   = "#1f3a5f"
MAROON = "#8b1a1a"
FOREST = "#2d6a2d"
AMBER  = "#b8860b"
GREY   = "#888888"

# ── GDP level estimation (src/gdp_estimation.py) ─────────────
GDP_EST_TRAIN_END = 2017  # train on years up to and including this
GDP_EST_YEARS = [2018, 2019, 2020]
COVID_YEAR = 2020

# The synthetic GDP generator has no recession in it. In synthetic mode this
# injects a 2020 contraction into GDP while leaving lights unchanged, which
# mimics the real phenomenon (activity fell faster than light). No effect when
# USE_SYNTHETIC is False, since real data carries COVID natively.
SYNTHETIC_COVID_DEMO = True
SYNTHETIC_COVID_SHOCK = 0.07

# ── Data-poor country application (src/country_application.py) ─
COUNTRY_USE_SYNTHETIC = True  # swap to real WDI + country NTL when False
TARGET_COUNTRY = "Example Republic"
COUNTRY_ANCHOR_YEAR = 2015  # the one year a GDP figure is assumed known
COUNTRY_YEARS = list(range(2013, 2023))
COUNTRY_ELASTICITY_LOW = 0.20  # plausible elasticity range for the
COUNTRY_ELASTICITY_HIGH = 0.45  # sensitivity band

COUNTRY_YEARS = list(range(2013, 2023))
COUNTRY_ANCHOR_YEAR = 2015  # the one target-country GDP year treated as known
COUNTRY_ELASTICITY_LOW = 0.20  # sensitivity band
COUNTRY_ELASTICITY_HIGH = 0.45

# Real sources
GDP_INDICATOR = "NY.GDP.MKTP.KD"  # GDP, constant 2015 US$ (World Bank WDI)
NASA_TOKEN_ENV = "BLACKMARBLE_TOKEN"  # env var holding your NASA Earthdata token
TARGET_ISO3 = "GHA"  # Ghana (has WDI data to validate against)
MAP_YEAR = 2021  # year drawn on the maps
PANEL_MAX_COUNTRIES = 8  # cap on data-rich countries (bounds NTL download)
PANEL_INCOME_GROUPS = ["High income", "Upper middle income"]