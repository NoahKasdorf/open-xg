
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss

# 1. Load StatsBomb shot data
df = pd.read_csv("data/processed/shots.csv")

# Keep rows with known goal outcome and match ID
df = df.dropna(subset=["goal", "match_id"]).copy()

# 2. Select the same features as Logistic Regression
numeric = ["distance", "angle"]
categorical = ["body_part", "technique", "under_pressure"]

X = df[numeric + categorical].copy()
y = df["goal"].astype(int)

# 3. Split matches into training and testing sets
matches = df["match_id"].unique()

train_matches, test_matches = train_test_split(
    matches,
    test_size=0.2,
    random_state=42
)

train_mask = df["match_id"].isin(train_matches)
test_mask = df["match_id"].isin(test_matches)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

groups_train = df.loc[train_mask, "match_id"]

print("Training shots:", len(X_train))
print("Testing shots:", len(X_test))

# 4. Prepare the data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocess = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric),
    ("categorical", categorical_pipeline, categorical)
])

# 5. Create Gradient Boosting model
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=False
    ))
])

# 6. Hyperparameter tuning
parameters = {
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__max_leaf_nodes": [15, 31],
    "classifier__max_iter": [100, 200]
}

# Grouped cross-validation by match
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
    groups=groups_train
)

# 7. Select the best model
best_model = search.best_estimator_

print("Best hyperparameters:", search.best_params_)
print("Best CV Log Loss:", round(-search.best_score_, 4))

# 8. Predict on the test set
predictions = best_model.predict_proba(X_test)[:, 1]

print("Training complete!")
print("Gradient Boosting Log Loss:",
      round(log_loss(y_test, predictions), 4))
print("Gradient Boosting Brier Score:",
      round(brier_score_loss(y_test, predictions), 4))

# 9. Compare with StatsBomb xG when available
provider_xg = pd.to_numeric(
    df.loc[test_mask, "statsbomb_xg"],
    errors="coerce"
)

valid = provider_xg.notna() & provider_xg.between(0, 1)

if valid.any():
    print("StatsBomb Brier Score:", round(
        brier_score_loss(
            y_test.loc[valid],
            provider_xg.loc[valid]
        ), 4
    ))

# 10. Save the trained model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "models/gradient_boosting.joblib"
)

print("Model saved to models/gradient_boosting.joblib")
