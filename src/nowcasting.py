import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from src.config import OUTPUT_TAB, TRAIN_END_YEAR, TEST_START_YEAR, YEARS

log = logging.getLogger(__name__)


def run_nowcasting(panel):
    log.info(f"Nowcasting: train <={TRAIN_END_YEAR}, test >={TEST_START_YEAR}")

    # Use growth rates — predict GDP growth from NTL growth
    df = panel.dropna(subset=["gdp_growth","ntl_growth"]).copy()

    train = df[df["year"] <= TRAIN_END_YEAR]
    test  = df[df["year"] >= TEST_START_YEAR]

    if len(train) < 100 or len(test) < 50:
        log.warning("Insufficient data for nowcasting split.")
        return {}

    features = ["ntl_growth","ln_pop","ln_ntl"]
    features = [f for f in features if f in df.columns]

    X_train = train[features].values
    y_train = train["gdp_growth"].values
    X_test  = test[features].values
    y_test  = test["gdp_growth"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Model 1: Ridge regression nowcast
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)

    # Model 2: Naive benchmark (predict mean growth = 0)
    pred_naive = np.zeros(len(y_test))

    rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
    rmse_naive = np.sqrt(mean_squared_error(y_test, pred_naive))
    mae_ridge  = mean_absolute_error(y_test, pred_ridge)
    mae_naive  = mean_absolute_error(y_test, pred_naive)

    results = {
        "model":      ["NTL Nowcast (Ridge)", "Naive (zero growth)"],
        "RMSE":       [round(rmse_ridge, 4), round(rmse_naive, 4)],
        "MAE":        [round(mae_ridge, 4),  round(mae_naive, 4)],
        "R2_test":    [round(float(ridge.score(X_test, y_test)), 4), 0.0],
    }
    results_df = pd.DataFrame(results).set_index("model")
    results_df.to_csv(OUTPUT_TAB / "nowcasting_results.csv")

    log.info("\nNowcasting Results:")
    log.info(results_df.to_string())
    log.info(f"\nNTL reduces RMSE by "
             f"{(rmse_naive - rmse_ridge)/rmse_naive*100:.1f}% vs naive benchmark")

    # Save predictions for plotting
    test_out = test.copy()
    test_out["gdp_growth_pred"] = pred_ridge
    test_out.to_csv(OUTPUT_TAB / "nowcast_predictions.csv", index=False)

    return {"ridge": ridge, "metrics": results_df,
            "predictions": test_out, "scaler": scaler}