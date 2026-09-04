import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# EXACT features for website/model
features = [
    'OverallQual',
    'GrLivArea',
    'GarageArea',
    '1stFlrSF',
    'FullBath',
    'YearBuilt',
    'YearRemodAdd',
    'MasVnrArea',
    'Fireplaces',
    'BsmtFinSF1',
    'LotFrontage',
    'WoodDeckSF',
    'OpenPorchSF',
    'LotArea',
    'CentralAir'
]

target = "SalePrice"

# Keep only required columns
df = df[features + [target]].copy()

# CentralAir: Y/N -> 1/0
df["CentralAir"] = df["CentralAir"].map({
    "Y": 1,
    "N": 0
})

# Convert columns to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[target] = pd.to_numeric(df[target], errors="coerce")

# Fill missing values
for col in features:
    df[col] = df[col].fillna(df[col].median())

# Remove missing target rows
df = df.dropna(subset=[target])

# X and y
X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.01,
    max_depth=6,
    n_estimators=1000,
    subsample=0.65,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save NEW model
joblib.dump(model, "xgb_model.jb")

print("Model saved successfully!")
print("Features:")
print(features)