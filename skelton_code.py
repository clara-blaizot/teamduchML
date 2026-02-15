import geopandas as gpd
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# =====================
# 1. Mapping des classes
# =====================
change_type_map = {
    'Demolition': 0,
    'Road': 1,
    'Residential': 2,
    'Commercial': 3,
    'Industrial': 4,
    'Mega Projects': 5
}

# =====================
# 2. Lecture des données
# =====================
train_df = gpd.read_file("train.geojson")
test_df = gpd.read_file("test.geojson")

# =====================
# 3. Reprojection
# =====================
# Permet d'avoir des aires et périmètres en mètres
train_df = train_df.to_crs(epsg=3857)
test_df = test_df.to_crs(epsg=3857)

# =====================
# 4. Feature engineering
# =====================
def extract_features(gdf):
    features = pd.DataFrame(index=gdf.index)

    # ---- Géométrie
    # Peut produire NaN ou inf → géré plus tard par la médiane
    features["area"] = gdf.geometry.area
    features["perimeter"] = gdf.geometry.length

    # ---- RGB date 1
    features["red_mean_d1"] = gdf["img_red_mean_date1"]
    features["green_mean_d1"] = gdf["img_green_mean_date1"]
    features["blue_mean_d1"] = gdf["img_blue_mean_date1"]

    # ---- RGB date 5
    features["red_mean_d5"] = gdf["img_red_mean_date5"]
    features["green_mean_d5"] = gdf["img_green_mean_date5"]
    features["blue_mean_d5"] = gdf["img_blue_mean_date5"]

    # ---- Différences RGB (évolution visuelle)
    features["red_diff"] = features["red_mean_d5"] - features["red_mean_d1"]
    features["green_diff"] = features["green_mean_d5"] - features["green_mean_d1"]
    features["blue_diff"] = features["blue_mean_d5"] - features["blue_mean_d1"]

    # ---- Temps (durée du projet en jours)
    d1 = pd.to_datetime(gdf["date1"], dayfirst=True, errors="coerce")
    d5 = pd.to_datetime(gdf["date4"], dayfirst=True, errors="coerce")
    duration = (d5 - d1).dt.days

    # Durées négatives → NaN
    duration = duration.where(duration >= 0)
    features["duration_days"] = duration

    # ---- Étape CRUCIALE : inf → NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    return features

# =====================
# 5. Construction X / y
# =====================
X_train = extract_features(train_df)
X_test = extract_features(test_df)

y_train = train_df["change_type"].map(change_type_map)

print("Train X :", X_train.shape)
print("Train y :", y_train.shape)
print("Test X  :", X_test.shape)

# =====================
# 6. Pipeline ML
# =====================
pipeline = Pipeline([
    # Remplacement de toutes les valeurs manquantes par la médiane
    ("imputer", SimpleImputer(strategy="median")),

    # Modèle final
    ("gb", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# =====================
# 7. Entraînement
# =====================
pipeline.fit(X_train, y_train)

# =====================
# 8. Prédiction
# =====================
pred_y = pipeline.predict(X_test)
print("Predictions :", pred_y.shape)

# =====================
# 9. Fichier de soumission Kaggle
# =====================
submission = pd.DataFrame({
    "Id": X_test.index,
    "change_type": pred_y
})

submission.to_csv("gradient_boosting_submission.csv", index=False)
print("✅ Fichier gradient_boosting_submission.csv créé")
