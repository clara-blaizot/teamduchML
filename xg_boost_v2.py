import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2, 
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}

print("="*70)
print("MODÈLE XGBOOST - ÉTAPE 1")
print("="*70)

# CHARGEMENT
print("\n[1/7] Chargement des données...")
train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

# FEATURES GÉOMÉTRIQUES
print("[2/7] Extraction features géométriques...")
train_df['area'] = train_df.geometry.area
train_df['perimeter'] = train_df.geometry.length
train_df['compactness'] = 4 * np.pi * train_df['area'] / (train_df['perimeter'] ** 2)

test_df['area'] = test_df.geometry.area
test_df['perimeter'] = test_df.geometry.length
test_df['compactness'] = 4 * np.pi * test_df['area'] / (test_df['perimeter'] ** 2)

# SÉLECTION FEATURES
print("[3/7] Sélection des features...")
image_features = [col for col in train_df.columns if col.startswith('img_')]
geom_features = ['area', 'perimeter', 'compactness']
temporal_features = [col for col in train_df.columns if 'change_status' in col]
categorical_features = ['urban_type', 'geography_type']
numeric_features = geom_features + image_features + temporal_features

# ENCODE TEMPOREL
for col in temporal_features:
    le = {}
    unique_vals = sorted(set(train_df[col].dropna().unique()) | set(test_df[col].dropna().unique()))
    for i, val in enumerate(unique_vals):
        le[val] = i
    train_df[col] = train_df[col].map(le).fillna(-1)
    test_df[col] = test_df[col].map(le).fillna(-1)

# PREPROCESSING
print("[4/7] Preprocessing...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

X_train = preprocessor.fit_transform(train_df[numeric_features + categorical_features])
y_train = train_df['change_type'].map(change_type_map).values
X_test = preprocessor.transform(test_df[numeric_features + categorical_features])

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# ENTRAINEMENT XGBOOST
print("[5/7] Entraînement XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    reg_alpha=0.5,
    min_child_weight=1,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(X_train_split, y_train_split)

# ÉVALUATION
print("[6/7] Évaluation...")
val_pred = xgb_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"\n✓ Accuracy XGBoost sur validation: {val_accuracy*100:.2f}%")
print(classification_report(y_val, val_pred, target_names=list(change_type_map.keys()), digits=3))

# PRÉDICTIONS
print("[7/7] Génération prédictions test...")
y_test_pred = xgb_model.predict(X_test)

pred_df = pd.DataFrame({'change_type': y_test_pred})
pred_df.to_csv("xgboost_submission.csv", index=True, index_label='Id')

print("\n" + "="*70)
print("✓ FICHIER SAUVEGARDÉ: xgboost_submission.csv")
print("="*70)