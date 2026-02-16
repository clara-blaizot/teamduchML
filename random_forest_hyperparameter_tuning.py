# ======================
# IMPORTS
# ======================
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======================
# CONFIG
# ======================
NUMERIC_FEATURES_EXTRA = ['area', 'perimeter', 'compactness']  # add more if needed
CATEGORICAL_FEATURES = ['urban_type', 'geography_type']

QUICK_TREES = 150      # Small number of trees for quick tuning
FINAL_TREES = 600      # Final model
MIN_VALID_ACC = 0.945  # Example threshold to proceed to full training

# ======================
# LOAD DATA
# ======================
train_df = gpd.read_file('train.geojson', index_col=0)
test_df  = gpd.read_file('test.geojson', index_col=0)

# Geometry features
for df in [train_df, test_df]:
    df['area'] = df.geometry.area
    df['perimeter'] = df.geometry.length
    df['compactness'] = 4 * np.pi * df['area'] / (df['perimeter'] ** 2)

# Numeric features (add image stats if needed)
numeric_features = NUMERIC_FEATURES_EXTRA + [c for c in train_df.columns if c.startswith('img_')]

# Target mapping
change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2,
                   'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5}
y = train_df['change_type'].map(change_type_map)

# ======================
# PREPROCESSING
# ======================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), CATEGORICAL_FEATURES)
    ]
)

X = preprocessor.fit_transform(train_df[numeric_features + CATEGORICAL_FEATURES])
X_test = preprocessor.transform(test_df[numeric_features + CATEGORICAL_FEATURES])

# ======================
# QUICK VALIDATION SPLIT
# ======================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ======================
# 1️⃣ QUICK TUNING MODEL
# ======================
print(f"Training quick Random Forest with {QUICK_TREES} trees...")
rf_quick = RandomForestClassifier(n_estimators=QUICK_TREES, random_state=42, n_jobs=-1)
rf_quick.fit(X_train, y_train)

y_val_pred = rf_quick.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy with {QUICK_TREES} trees: {val_acc:.4f}")

# ======================
# 2️⃣ FINAL MODEL IF PROMISING
# ======================
if val_acc >= MIN_VALID_ACC:
    print(f"Validation accuracy ≥ {MIN_VALID_ACC}, training final model with {FINAL_TREES} trees...")
    rf_final = RandomForestClassifier(n_estimators=FINAL_TREES, random_state=42, n_jobs=-1)
    rf_final.fit(X, y)  # Train on full data
    
    predictions = rf_final.predict(X_test)
    submission = pd.DataFrame({
        "Id": range(len(predictions)),
        "change_type": predictions
    })
    submission.to_csv("rf_final_submission.csv", index=False)
    print(f"Final submission saved! Shape: {submission.shape}")
else:
    print("Validation accuracy too low, adjust hyperparameters before final training.")
