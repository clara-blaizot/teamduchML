import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2, 
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}

print("="*70)
print("ENSEMBLE VOTING (XGB + LGB + CATBOOST) - POUR 97%")
print("="*70)

# CHARGEMENT
print("\n[1/10] Chargement des données...")
train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

# FEATURES GÉOMÉTRIQUES
print("[2/10] Extraction features géométriques...")
train_df['area'] = train_df.geometry.area
train_df['perimeter'] = train_df.geometry.length
train_df['compactness'] = 4 * np.pi * train_df['area'] / (train_df['perimeter'] ** 2)

test_df['area'] = test_df.geometry.area
test_df['perimeter'] = test_df.geometry.length
test_df['compactness'] = 4 * np.pi * test_df['area'] / (test_df['perimeter'] ** 2)

# SÉLECTION FEATURES
print("[3/10] Sélection des features...")
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
print("[4/10] Preprocessing...")
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

# ========== MODÈLE 1 : XGBOOST ==========
print("[5/10] Entraînement XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    reg_alpha=0.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(X_train_split, y_train_split)

xgb_pred = xgb_model.predict(X_val)
xgb_acc = accuracy_score(y_val, xgb_pred)
print(f"   XGBoost accuracy: {xgb_acc*100:.2f}%")

# ========== MODÈLE 2 : LIGHTGBM ==========
print("[6/10] Entraînement LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=9,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.3,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lgb_model.fit(X_train_split, y_train_split)

lgb_pred = lgb_model.predict(X_val)
lgb_acc = accuracy_score(y_val, lgb_pred)
print(f"   LightGBM accuracy: {lgb_acc*100:.2f}%")

# ========== MODÈLE 3 : CATBOOST ==========
print("[7/10] Entraînement CatBoost...")
cb_model = cb.CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.09,
    l2_leaf_reg=2.0,
    random_state=42,
    thread_count=-1,
    verbose=False
)

cb_model.fit(X_train_split, y_train_split)

cb_pred = cb_model.predict(X_val)
cb_acc = accuracy_score(y_val, cb_pred)
print(f"   CatBoost accuracy: {cb_acc*100:.2f}%")

# ========== ENSEMBLE VOTING ==========
print("[8/10] Création et entraînement de l'ensemble voting...")
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cb', cb_model)
    ],
    voting='soft'
)

voting_clf.fit(X_train_split, y_train_split)

# ÉVALUATION ENSEMBLE
print("[9/10] Évaluation ensemble...")
ensemble_pred = voting_clf.predict(X_val)
ensemble_acc = accuracy_score(y_val, ensemble_pred)

print(f"\n{'RÉSULTATS VALIDATION':<40} {'ACCURACY':<15}")
print("-" * 55)
print(f"{'XGBoost':<40} {xgb_acc*100:>6.2f}%")
print(f"{'LightGBM':<40} {lgb_acc*100:>6.2f}%")
print(f"{'CatBoost':<40} {cb_acc*100:>6.2f}%")
print(f"{'ENSEMBLE VOTING (SOFT)':<40} {ensemble_acc*100:>6.2f}%")
print("-" * 55)

print("\nRapport de classification ensemble:")
print(classification_report(y_val, ensemble_pred, 
                          target_names=list(change_type_map.keys()), digits=3))

# PRÉDICTIONS TEST
print("[10/10] Génération prédictions test...")
y_test_pred = voting_clf.predict(X_test)

pred_df = pd.DataFrame({'change_type': y_test_pred})
pred_df.to_csv("ensemble_submission.csv", index=True, index_label='Id')

print("\n" + "="*70)
print("✓ FICHIER SAUVEGARDÉ: ensemble_submission.csv")
print("="*70)