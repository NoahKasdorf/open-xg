
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
import joblib

# Load StatsBomb shot data
df = pd.read_csv("data/processed/shots.csv")

numeric = ["distance", "angle"]
categorical = ["body_part", "technique", "under_pressure"]

# Use only rows with a known goal outcome
df = df.dropna(subset=["goal"])
X = df[numeric + categorical].copy()
y = df["goal"].astype(int)

# Split by match to avoid sharing matches between sets
matches = df["match_id"].dropna().unique()
train_matches, test_matches = train_test_split(
    matches, test_size=0.2, random_state=42
)

train_mask = df["match_id"].isin(train_matches)
test_mask = df["match_id"].isin(test_matches)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric),
    ("categorical", categorical_pipeline, categorical)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000
    ))
])

# Train the model
model.fit(X.loc[train_mask], y.loc[train_mask])

# Predict goal probabilities
predictions = model.predict_proba(
    X.loc[test_mask]
)[:, 1]

actual = y.loc[test_mask]

print("Training complete!")
print("Training shots:", train_mask.sum())
print("Testing shots:", test_mask.sum())
print("Log Loss:", round(log_loss(actual, predictions), 4))
print("Brier Score:", round(
    brier_score_loss(actual, predictions), 4
))

# Compare against StatsBomb xG when available
provider_xg = pd.to_numeric(
    df.loc[test_mask, "statsbomb_xg"],
    errors="coerce"
)
valid = provider_xg.notna() & provider_xg.between(0, 1)

if valid.any():
    print("StatsBomb Brier Score:", round(
        brier_score_loss(
            actual.loc[valid],
            provider_xg.loc[valid]
        ), 4
    ))

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/logistic_l2.joblib")
print("Model saved to models/logistic_l2.joblib")
# Show model coefficients
feature_names = model.named_steps["preprocess"].get_feature_names_out()
coefficients = model.named_steps["classifier"].coef_[0]

coef_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

print(coef_table.sort_values("Coefficient"))