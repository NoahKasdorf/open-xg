
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.dataset import load_dataset, save_predictions

Path("models").mkdir(exist_ok=True)

for feature_set in ["geometry", "base", "base_ff"]:
    print(f"\nRunning Logistic Regression: {feature_set}")

    X_train, y_train, _ = load_dataset(
        "train", feature_set=feature_set
    )
    X_valid, y_valid, meta_valid = load_dataset(
        "valid", feature_set=feature_set
    )
    X_test, _, meta_test = load_dataset(
        "test", feature_set=feature_set
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train the model on train only
    model.fit(X_train, y_train)

    # Fit Platt calibration on valid only
    valid_raw = model.predict_proba(X_valid)[:, 1]

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(valid_raw.reshape(-1, 1), y_valid)

    # Calibrated predictions for valid and test
    valid_calibrated = calibrator.predict_proba(
        valid_raw.reshape(-1, 1)
    )[:, 1]

    test_raw = model.predict_proba(X_test)[:, 1]
    test_calibrated = calibrator.predict_proba(
        test_raw.reshape(-1, 1)
    )[:, 1]

    valid_calibrated = np.clip(
        valid_calibrated, 1e-6, 1 - 1e-6
    )
    test_calibrated = np.clip(
        test_calibrated, 1e-6, 1 - 1e-6
    )

    # Save predictions separately for evaluation
    save_predictions(
        meta_valid["shot_id"],
        valid_calibrated,
        f"lr_{feature_set}_platt_valid"
    )
    save_predictions(
        meta_test["shot_id"],
        test_calibrated,
        f"lr_{feature_set}_platt_test"
    )

    # Save trained model and calibrator
    joblib.dump(model, f"models/lr_{feature_set}.joblib")
    joblib.dump(
        calibrator,
        f"models/lr_{feature_set}_platt.joblib"
    )

    print(f"Completed: {feature_set}")

print("\nAll Logistic Regression runs completed!")
