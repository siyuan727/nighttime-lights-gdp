import pandas as pd
import numpy as np
import logging
from src.config import OUTPUT_TAB, STATE_SHP_PATH

log = logging.getLogger(__name__)


def build_state_weights(state_gdf):
    from libpysal.weights import Queen
    log.info("Building spatial weights matrix...")
    w = Queen.from_dataframe(state_gdf, use_index=False)
    # Handle islands (states with no queen neighbors)
    if w.n_components > 1:
        log.warning(f"{w.n_components} components in spatial weights — "
                    "islands present. Results may be affected.")
    w.transform = "r"   # row-standardise
    log.info(f"Weights: {w.n} units, avg {w.mean_neighbors:.1f} neighbors")
    return w


def run_spatial_lag(state_panel, state_gdf):
    try:
        from spreg import GM_Lag
        import libpysal
    except ImportError:
        raise ImportError("Install with: pip install spreg libpysal")

    log.info("Running spatial lag model...")

    # Use most recent year for cross-sectional spatial model
    latest = state_panel[state_panel["year"] == state_panel["year"].max()].copy()

    # Align GDF and panel on state_fips
    state_gdf = state_gdf.copy()
    state_gdf["state_fips"] = state_gdf["STATEFIP"] if "STATEFIP" in state_gdf.columns \
        else state_gdf["STATEFP"]

    from src.config import EXCLUDE_FIPS_PREFIX
    state_gdf = state_gdf[~state_gdf["state_fips"].isin(EXCLUDE_FIPS_PREFIX)]
    merged = state_gdf.merge(latest, on="state_fips", how="inner")
    merged = merged.dropna(subset=["ln_gdp","ln_ntl","ln_pop"])
    merged = merged.reset_index(drop=True)

    w = build_state_weights(merged)

    y = merged["ln_gdp"].values.reshape(-1, 1)
    X = np.column_stack([
        np.ones(len(merged)),
        merged["ln_ntl"].values,
        merged["ln_pop"].values,
    ])

    model = GM_Lag(
        y, X, w=w,
        name_y="ln_gdp",
        name_x=["const", "ln_ntl", "ln_pop"],
        name_ds="US States"
    )
    log.info("\n" + str(model.summary))

    # Save results
    rows = []
    for i, name in enumerate(["const","ln_ntl","ln_pop"]):
        rows.append({"variable": name,
                     "coef": round(float(model.betas[i]), 4),
                     "std_err": round(float(model.std_err[i]), 4),
                     "z_stat": round(float(model.z_stat[i][0]), 4),
                     "pval":   round(float(model.z_stat[i][1]), 4)})
    # Spatial lag coefficient (rho)
    rows.append({"variable": "rho (spatial lag)",
                 "coef": round(float(model.betas[-1]), 4),
                 "std_err": round(float(model.std_err[-1]), 4),
                 "z_stat": round(float(model.z_stat[-1][0]), 4),
                 "pval":   round(float(model.z_stat[-1][1]), 4)})
    pd.DataFrame(rows).to_csv(OUTPUT_TAB / "spatial_model.csv", index=False)
    log.info("Spatial model results saved.")
    return model, merged