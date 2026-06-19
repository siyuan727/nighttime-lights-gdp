import pandas as pd
import numpy as np
import logging
from src.config import DATA_CLEAN, PANEL_PKL, STATE_PANEL_PKL, YEARS, MIN_YEARS_REQUIRED, EXCLUDE_FIPS_PREFIX, START_YEAR

log = logging.getLogger(__name__)


def build_county_panel(gdp_df, ntl_df, pop_df, save=True):
    log.info("Building county panel...")
    for df in [gdp_df, ntl_df, pop_df]:
        df["fips"] = df["fips"].astype(str).str.zfill(5)

    panel = gdp_df.merge(ntl_df, on=["fips","year"], how="inner")
    panel = panel.merge(pop_df[["fips","year","population"]],
                        on=["fips","year"], how="left")

    panel = panel[~panel["fips"].str[:2].isin(EXCLUDE_FIPS_PREFIX)]
    panel = panel[(panel["gdp_thousands"] > 0) & (panel["ntl_mean"] > 0)]

    panel["population"] = (panel.groupby("fips")["population"]
                           .transform(lambda x: x.fillna(x.mean())))
    panel["population"] = (panel.groupby("year")["population"]
                           .transform(lambda x: x.fillna(x.median())))
    panel["population"] = panel["population"].fillna(10000).astype(int)

    panel["ln_gdp"]    = np.log(panel["gdp_thousands"])
    panel["ln_ntl"]    = np.log(panel["ntl_mean"] + 0.01)
    panel["ln_pop"]    = np.log(panel["population"].clip(1))
    panel["gdp_pc"]    = (panel["gdp_thousands"] * 1000) / panel["population"].clip(1)
    panel["ln_gdp_pc"] = np.log(panel["gdp_pc"].clip(1))
    panel["state_fips"] = panel["fips"].str[:2]
    panel["urban"]      = (panel["population"] > 100_000).astype(int)

    panel = panel.sort_values(["fips","year"])
    panel["gdp_growth"] = panel.groupby("fips")["ln_gdp"].diff()
    panel["ntl_growth"] = panel.groupby("fips")["ln_ntl"].diff()

    valid = panel.groupby("fips")["year"].count()
    panel = panel[panel["fips"].isin(valid[valid >= MIN_YEARS_REQUIRED].index)]
    panel = panel.reset_index(drop=True)

    log.info(f"Panel: {len(panel)} rows, {panel['fips'].nunique()} counties, "
             f"{panel['state_fips'].nunique()} states")
    log.info(f"NTL-GDP corr: {panel[['ln_gdp','ln_ntl']].corr().iloc[0,1]:.4f}")

    if save:
        DATA_CLEAN.mkdir(parents=True, exist_ok=True)
        panel.to_pickle(PANEL_PKL)
        panel.to_csv(DATA_CLEAN / "panel_county.csv", index=False)
    return panel


def build_state_panel(county_panel, save=True):
    log.info("Building state panel...")
    state = county_panel.groupby(["state_fips","year"]).agg(
        gdp_thousands=("gdp_thousands", "sum"),
        ntl_mean=("ntl_mean", lambda x:
                  np.average(x, weights=county_panel.loc[x.index,"population"])),
        population=("population", "sum"),
        n_counties=("fips", "count"),
    ).reset_index()

    state["ln_gdp"]    = np.log(state["gdp_thousands"])
    state["ln_ntl"]    = np.log(state["ntl_mean"] + 0.01)
    state["ln_pop"]    = np.log(state["population"].clip(1))
    state["gdp_pc"]    = (state["gdp_thousands"] * 1000) / state["population"].clip(1)
    state["ln_gdp_pc"] = np.log(state["gdp_pc"].clip(1))
    state = state.sort_values(["state_fips","year"])
    state["gdp_growth"] = state.groupby("state_fips")["ln_gdp"].diff()
    state["ntl_growth"] = state.groupby("state_fips")["ln_ntl"].diff()

    if save:
        state.to_pickle(STATE_PANEL_PKL)
        state.to_csv(DATA_CLEAN / "panel_state.csv", index=False)
    return state


def descriptive_statistics(panel):
    cols = ["ln_gdp","ln_ntl","ln_pop","gdp_growth","ntl_growth"]
    cols = [c for c in cols if c in panel.columns]
    stats = panel[cols].describe().T
    stats["skew"]     = panel[cols].skew()
    stats["kurtosis"] = panel[cols].kurtosis()
    log.info("\nDescriptive Statistics:\n" + stats.round(3).to_string())
    return stats