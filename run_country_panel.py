"""
Developing-economies external-validity analysis (the Ghana replacement).

Runs the representative cross-country exercise: learn the light-GDP elasticity on
high-income countries, reconstruct GDP for every developing economy with complete
data, and regress reconstruction error on structural covariates (agriculture
share, electricity access, rural share, informality).

This is separated from run_improvements.py because it downloads national Black
Marble lights for ~150 countries, which is the slow step. It caches, so later
runs are fast. Set REFERENCE_MAX_COUNTRIES / DEV_MAX_COUNTRIES in config to small
integers for a quick first pass, then None for the full panel.

Needs: BLACKMARBLE_TOKEN set, wbgapi, geopandas, blackmarblepy installed (the
same stack the old country module used).
"""

import logging

from src.config import OUTPUT_TAB, OUTPUT_FIG, OUTPUT_LOG

for d in (OUTPUT_TAB, OUTPUT_FIG, OUTPUT_LOG):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(OUTPUT_LOG / "country_panel.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("country_panel_run")


def main():
    from src.country_panel import run, plot_error_vs_covariate
    res = run(save=True)
    # Scatter the reconstruction error against each structural covariate.
    for cov in ("agriculture_share", "electricity_access",
                "rural_share", "self_employment"):
        try:
            plot_error_vs_covariate(res["errors"], covariate=cov)
        except Exception as e:
            log.warning(f"Plot for {cov} skipped: {e}")
    s = res["summary"]
    log.info(f"Done. Reference beta {s['reference_beta']:.3f} on "
             f"{s['n_reference']} countries; {s['n_developing']} developing "
             f"economies; mean reconstruction MAPE {s['mean_mape']:.1f}%.")


if __name__ == "__main__":
    main()
