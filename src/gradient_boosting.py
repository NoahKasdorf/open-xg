
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV

from src.dataset import load_dataset, save_predictions

Path("models").mkdir(exist_ok=True)

feature_sets = ["geometry", "base", "base_ff"]

parameters = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "max_iter": [100, 200],
}

for feature_set in feature_sets:
    print(f"\nRunning Gradient Boosting: {feature_set}")

    X_train, y_train, meta_train = load_dataset(
        "train", feature_set=feature_set
    )
    X_valid, y_valid, meta_valid = load_dataset(
        "valid", feature_set=feature_set
    )
    X_test, _, meta_test = load_dataset(
        "test", feature_set=feature_set
    )

    model = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=False
    )

    # Tune hyperparameters using train only
    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="neg_log_loss",
        cv=GroupKFold(n_splits=3),
        n_jobs=-1,
        verbose=1
    )

    search.fit(
        X_train,
        y_train,
        groups=meta_train["match_id"]
    )

    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)

    # Fit Platt calibration on valid only
    valid_raw = best_model.predict_proba(X_valid)[:, 1]

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(
        valid_raw.reshape(-1, 1),
        y_valid
    )

    valid_calibrated = calibrator.predict_proba(
        valid_raw.reshape(-1, 1)
    )[:, 1]

    test_raw = best_model.predict_proba(X_test)[:, 1]
    test_calibrated = calibrator.predict_proba(
        test_raw.reshape(-1, 1)
    )[:, 1]

    valid_calibrated = np.clip(
        valid_calibrated, 1e-6, 1 - 1e-6
    )
    test_calibrated = np.clip(
        test_calibrated, 1e-6, 1 - 1e-6
    )

    # Save predictions for team evaluation
    save_predictions(
        meta_valid["shot_id"],
        valid_calibrated,
        f"gbm_{feature_set}_platt_valid"
    )
    save_predictions(
        meta_test["shot_id"],
        test_calibrated,
        f"gbm_{feature_set}_platt_test"
    )

    # Save model and calibrator
    joblib.dump(
        best_model,
        f"models/gbm_{feature_set}.joblib"
    )
    joblib.dump(
        calibrator,
        f"models/gbm_{feature_set}_platt.joblib"
    )

    print(f"Completed: {feature_set}")

print("\nAll Gradient Boosting runs completed!")
