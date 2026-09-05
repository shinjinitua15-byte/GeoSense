# train_model.py
# Purpose: Train and compare Random Forest and XGBoost
#          for site suitability and explain the best model with SHAP.

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import shap


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

FEATURES = [
    "dist_road_m",
    "dist_school_m",
    "flood_risk"
]

TARGET = "label"


def train_and_evaluate():

    # -----------------------------------------------------
    # 1. Load labelled dataset
    # -----------------------------------------------------

    df = pd.read_csv(
        "data/processed/labelled_sites.csv"
    )

    print("Total labelled locations:", len(df))

    # -----------------------------------------------------
    # 2. Prepare X and y
    # -----------------------------------------------------

    X = df[FEATURES]
    y = df[TARGET]

    # -----------------------------------------------------
    # 3. Stratified 80/20 train-test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"Training on {len(X_train)} sites, "
        f"testing on {len(X_test)}"
    )

    # -----------------------------------------------------
    # 4. Random Forest
    # -----------------------------------------------------

    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    rf_score = rf.score(
        X_test,
        y_test
    )

    print(
        f"Random Forest Accuracy: {rf_score:.3f}"
    )

    print("\nRandom Forest Classification Report:")
    print(
        classification_report(
            y_test,
            rf.predict(X_test)
        )
    )

    # -----------------------------------------------------
    # 5. XGBoost
    # -----------------------------------------------------

    print("\nTraining XGBoost...")

    xg = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss"
    )

    xg.fit(X_train, y_train)

    xg_score = xg.score(
        X_test,
        y_test
    )

    print(
        f"XGBoost Accuracy: {xg_score:.3f}"
    )

    print("\nXGBoost Classification Report:")
    print(
        classification_report(
            y_test,
            xg.predict(X_test)
        )
    )

    # -----------------------------------------------------
    # 6. Select the better model
    # -----------------------------------------------------

    if rf_score >= xg_score:
        best = rf
        best_name = "RandomForest"
    else:
        best = xg
        best_name = "XGBoost"

    print(
        f"\nBest model: {best_name}"
    )

    # -----------------------------------------------------
    # 7. SHAP explanation
    # -----------------------------------------------------

    print("\nGenerating SHAP explanation...")

    explainer = shap.TreeExplainer(best)
    shap_values = explainer.shap_values(X_test)

    # Handle different SHAP return formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[:, :, 1]

    # -----------------------------------------------------
    # 8. Save SHAP plot
    # -----------------------------------------------------

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=FEATURES,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/reports/shap_importance.png",
        bbox_inches="tight"
    )

    plt.close()

    print(
        "SHAP plot saved to "
        "outputs/reports/shap_importance.png"
    )

    # -----------------------------------------------------
    # 9. Save trained model
    # -----------------------------------------------------

    os.makedirs(
        "models/saved",
        exist_ok=True
    )

    joblib.dump(
        best,
        "models/saved/site_scorer_model.pkl"
    )

    print(
        "Model saved to "
        "models/saved/site_scorer_model.pkl"
    )


if __name__ == "__main__":
    train_and_evaluate()