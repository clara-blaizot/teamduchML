import geopandas as gpd
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Load data
train_df = gpd.read_file("train.geojson", index_col=0)
test_df  = gpd.read_file("test.geojson", index_col=0)

# Geometry features
for df in [train_df, test_df]:
    df["area"] = df.geometry.area
    df["perimeter"] = df.geometry.length
    df["compactness"] = 4 * np.pi * df["area"] / (df["perimeter"] ** 2)

# Target
change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2,
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}
y_train = train_df["change_type"].map(change_type_map)

# Features
numeric_features = (
    ["area", "perimeter", "compactness"] +
    [c for c in train_df.columns if c.startswith("img_")]
)
categorical_features = ["urban_type", "geography_type"]

X_train = train_df[numeric_features + categorical_features]
X_test  = test_df[numeric_features + categorical_features]

# CatBoost model
cat_model = CatBoostClassifier(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    random_seed=42,
    verbose=100
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=categorical_features
)

predictions = cat_model.predict(X_test).ravel()

submission = pd.DataFrame({
    "Id": range(len(predictions)),
    "change_type": predictions.astype(int)
})

submission.to_csv("catboost_submission.csv", index=False)
print("Submission shape:", submission.shape)
