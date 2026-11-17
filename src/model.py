
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

from src.dataset import load_dataset, save_predictions

# Load the team's official competition-based splits
X_train, y_train, meta_train = load_dataset("train", feature_set="base_ff")
X_valid, y_valid, meta_valid = load_dataset("valid", feature_set="base_ff")
X_test, y_test, meta_test = load_dataset("test", feature_set="base_ff")

# Logistic Regression pipeline
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

# Train only on the training split
model.fit(X_train, y_train)

# Evaluate on validation data
valid_predictions = model.predict_proba(X_valid)[:, 1]

print("Training complete!")
print("Training shots:", len(X_train))
print("Validation shots:", len(X_valid))
print("Validation Log Loss:", round(log_loss(y_valid, valid_predictions), 4))
print("Validation Brier Score:", round(brier_score_loss(y_valid, valid_predictions), 4))

# Save validation predictions for evaluation
valid_predictions = np.clip(valid_predictions, 1e-6, 1 - 1e-6)
save_predictions(meta_valid["shot_id"], valid_predictions, "lr_base_ff_valid")

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/logistic_l2.joblib")
print("Model saved to models/logistic_l2.joblib")
