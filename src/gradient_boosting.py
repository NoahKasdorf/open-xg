
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import log_loss, brier_score_loss

from src.dataset import load_dataset, save_predictions

# Load the team's official datasets
X_train, y_train, meta_train = load_dataset(
    "train", feature_set="base_ff"
)
X_valid, y_valid, meta_valid = load_dataset(
    "valid", feature_set="base_ff"
)

# Gradient Boosting model
model = HistGradientBoostingClassifier(
    random_state=42,
    early_stopping=False
)

# Hyperparameter tuning
parameters = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "max_iter": [100, 200]
}

# Keep shots from the same match together
cv = GroupKFold(n_splits=3)

search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    scoring="neg_log_loss",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("Training Gradient Boosting...")

search.fit(
    X_train,
    y_train,
    groups=meta_train["match_id"]
)

best_model = search.best_estimator_

print("Best hyperparameters:", search.best_params_)
print("Best CV Log Loss:", round(-search.best_score_, 4))

# Evaluate on validation data
predictions = best_model.predict_proba(X_valid)[:, 1]

print("Training complete!")
print("Training shots:", len(X_train))
print("Validation shots:", len(X_valid))
print("Validation Log Loss:", round(log_loss(y_valid, predictions), 4))
print("Validation Brier Score:", round(brier_score_loss(y_valid, predictions), 4))

# Save predictions for team evaluation
predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
save_predictions(
    meta_valid["shot_id"],
    predictions,
    "gbm_base_ff_valid"
)

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(best_model, "models/gradient_boosting.joblib")

print("Model saved to models/gradient_boosting.joblib")
