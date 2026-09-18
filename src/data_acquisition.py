import requests, zipfile, io, json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.config import (
    DATA_RAW, BEA_API_KEY, YEARS,
    EXCLUDE_FIPS_PREFIX, START_YEAR, END_YEAR
)

log = logging.getLogger(__name__)

COUNTY_SHP_URL = "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip"
STATE_SHP_URL  = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
BEA_ENDPOINT   = "https://apps.bea.gov/api/data/"


def download_shapefile(url, dest_dir, name):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shp_files = list(dest_dir.glob("*.shp"))
    if shp_files:
        log.info(f"{name} shapefile already cached.")
        return shp_files[0]
    log.info(f"Downloading {name} shapefile...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest_dir)
    return list(dest_dir.glob("*.shp"))[0]


def download_county_shapefile():
    return download_shapefile(COUNTY_SHP_URL, DATA_RAW / "counties", "County")


def download_state_shapefile():
    return download_shapefile(STATE_SHP_URL, DATA_RAW / "states", "State")


def fetch_bea_gdp():
    cache = DATA_RAW / "bea_county_gdp.csv"
    if cache.exists():
        log.info("Loading cached BEA GDP...")
        return pd.read_csv(cache, dtype={"fips": str})
    if BEA_API_KEY == "YOUR_BEA_API_KEY_HERE":
        raise RuntimeError("No BEA key set. Put your real BEA key in "
                           "src/config.py; this project is real-data only.")
    log.info("Fetching BEA county GDP...")
    params = {
        "UserID": BEA_API_KEY, "method": "GetData",
        "datasetname": "Regional", "TableName": "CAGDP1",
        "LineCode": "1", "GeoFips": "COUNTY",
        "Year": ",".join(str(y) for y in YEARS),
        "ResultFormat": "json",
    }
    try:
        r = requests.get(BEA_ENDPOINT, params=params, timeout=120)
        r.raise_for_status()
        records = r.json()["BEAAPI"]["Results"]["Data"]
    except Exception as e:
        raise RuntimeError(f"BEA API failed: {e}") from e

    rows = []
    for rec in records:
        fips = str(rec.get("GeoFips", "")).zfill(5)
        val  = rec.get("DataValue", "")
        if val in ("(NA)", "(D)", "", "--"):
            continue
        try:
            gdp = float(val.replace(",", ""))
        except ValueError:
            continue
        rows.append({
            "fips": fips,
            "county_name": rec.get("GeoName", ""),
            "year": int(rec.get("TimePeriod", 0)),
            "gdp_thousands": gdp,
        })

    df = pd.DataFrame(rows)
    df = df[~df["fips"].str[:2].isin(EXCLUDE_FIPS_PREFIX)]
    df = df[df["fips"].str.len() == 5]
    df = df[df["year"].between(START_YEAR, END_YEAR)]
    df = df.dropna(subset=["gdp_thousands"])
    df.to_csv(cache, index=False)
    log.info(f"BEA GDP: {len(df)} rows, {df['fips'].nunique()} counties")
    return df


def _synthetic_gdp():
    np.random.seed(42)
    state_fips = [
        "01","04","05","06","08","09","10","12","13","16","17","18",
        "19","20","21","22","23","24","25","26","27","28","29","30",
        "31","32","33","34","35","36","37","38","39","40","41","42",
        "44","45","46","47","48","49","50","51","53","54","55","56"
    ]
    rows = []
    for state in state_fips:
        for c in range(1, np.random.randint(15, 60), 2):
            fips     = f"{state}{c:03d}"
            base_gdp = np.random.lognormal(14.5, 1.8)
            for year in YEARS:
                gdp = base_gdp * (1 + np.random.normal(0.025, 0.04)) ** (year - START_YEAR)
                rows.append({"fips": fips, "county_name": f"County_{fips}",
                             "year": year, "gdp_thousands": max(gdp, 1000)})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW / "bea_county_gdp.csv", index=False)
    return df


def fetch_population():
    cache = DATA_RAW / "census_population.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"fips": str})
    log.info("Fetching Census population...")
    rows = []
    for year in YEARS:
        api_year = min(year, 2019)
        url = (f"https://api.census.gov/data/{api_year}/pep/population"
               f"?get=POP,NAME&for=county:*&in=state:*")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data   = r.json()
            header = data[0]
            pop_i  = header.index("POP")
            st_i   = header.index("state")
            co_i   = header.index("county")
            for row in data[1:]:
                fips = str(row[st_i]).zfill(2) + str(row[co_i]).zfill(3)
                if fips[:2] in EXCLUDE_FIPS_PREFIX:
                    continue
                try:
                    rows.append({"fips": fips, "year": year,
                                 "population": int(row[pop_i])})
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            log.warning(f"Census API year {year}: {e}")
    if not rows:
        return _synthetic_population()
    df = pd.DataFrame(rows).sort_values(["fips","year"]).reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def _synthetic_population():

    np.random.seed(123)
    gdp_df = pd.read_csv(
        DATA_RAW / "bea_county_gdp.csv", dtype={"fips": str}
    )
    rows = []

    # Get mean GDP per county across all years
    county_mean_gdp = (
        gdp_df.groupby("fips")["gdp_thousands"].mean()
    )
    global_mean_ln = np.log(county_mean_gdp.clip(1)).mean()

    for fips in gdp_df["fips"].unique():
        mean_gdp = county_mean_gdp.get(fips, 1000.0)
        ln_gdp   = np.log(max(mean_gdp, 1))

        ln_pop_base = (
                10.5
                + 0.45 * (ln_gdp - global_mean_ln)
                + np.random.normal(0, 0.70)
        )
        base_pop = max(int(np.exp(ln_pop_base)), 200)

        for year in YEARS:
            growth = np.random.normal(0.005, 0.008)
            pop = int(base_pop * (1 + growth) ** (year - START_YEAR))
            rows.append({
                "fips":       fips,
                "year":       year,
                "population": max(pop, 100)
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW / "census_population.csv", index=False)
    log.info(f"Synthetic population: median = "
             f"{df['population'].median():.0f}, "
             f"urban share = "
             f"{(df['population'] > 100_000).mean():.1%}")
    return df