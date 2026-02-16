# =========================
# Kaggle XGBoost Full Script
# - Feature engineering (geometry + temporal color)
# - Proper preprocessing (fit on train only)
# - Stratified CV (no leakage)
# - Final training on full train
# - Submission (string labels)
# =========================

import geopandas as gpd
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

import xgboost as xgb

# -------------------------
# 0) Config / label maps
# -------------------------
change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2,
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}
inv_change_type_map = {v: k for k, v in change_type_map.items()}
NUM_CLASSES = 6
RANDOM_STATE = 42

# -------------------------
# 1) Load data
# -------------------------
train_df = gpd.read_file('train.geojson', index_col=0)
test_df  = gpd.read_file('test.geojson', index_col=0)

# -------------------------
# 2) CRS sanity (IMPORTANT)
# If CRS is lat/lon (degrees), areas/perimeters are not meaningful.
# We'll project to Web Mercator if CRS is EPSG:4326.
# -------------------------
def ensure_metric_crs(gdf, epsg=3857):
    if gdf.crs is None:
        # If CRS is missing, we can't safely reproject.
        # You can set it manually if you know it's EPSG:4326:
        # gdf = gdf.set_crs(epsg=4326).to_crs(epsg=epsg)
        return gdf
    try:
        if str(gdf.crs).endswith("4326") or "EPSG:4326" in str(gdf.crs):
            return gdf.to_crs(epsg=epsg)
    except Exception:
        pass
    return gdf

train_df = ensure_metric_crs(train_df)
test_df  = ensure_metric_crs(test_df)

# -------------------------
# 3) Feature engineering
# -------------------------
def add_geometrical_features(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df.copy()

    # Basic geometry
    df['area'] = df.geometry.area
    df['perimeter'] = df.geometry.length
    df['compactness'] = 4 * np.pi * df['area'] / (df['perimeter'] ** 2 + 1e-6)

    # Bounding box features (GeoSeries.bounds is a DataFrame)
    b = df.geometry.bounds
    df['bbox_width']  = b.maxx - b.minx
    df['bbox_height'] = b.maxy - b.miny
    df['bbox_ratio']  = df['bbox_width'] / (df['bbox_height'] + 1e-6)

    # Elongation proxy
    df['elongation'] = df['perimeter'] / (2 * np.sqrt(np.pi * df['area']) + 1e-6)

    # Convexity
    df['convex_area'] = df.geometry.convex_hull.area
    df['convexity'] = df['area'] / (df['convex_area'] + 1e-6)

    # Vertex count (Polygon only)
    df['n_vertices'] = df.geometry.apply(
        lambda g: len(g.exterior.coords) if g is not None and g.geom_type == 'Polygon' else np.nan
    )

    # Interaction-ish ratio
    df['area_perimeter_ratio'] = df['area'] / (df['perimeter'] + 1e-6)

    return df


def compute_sorted_order_from_first_row(train_df: pd.DataFrame) -> list:
    # date0..date5 exist? keep those present
    date_cols = [f'date{i}' for i in range(6) if f'date{i}' in train_df.columns]
    if not date_cols:
        raise ValueError("No date0..date5 columns found in train_df.")

    parsed_dates = []
    for i, col in enumerate(date_cols):
        # dayfirst=True matches your example "06-07-2020"
        parsed_dates.append(pd.to_datetime(train_df[col].iloc[0], dayfirst=True, errors='coerce'))

    if any(pd.isna(d) for d in parsed_dates):
        raise ValueError("Could not parse some dates in first row. Check date format and dayfirst=True.")

    # sorted_order contains the indices (0..len(date_cols)-1) corresponding to date0..date5 positions
    sorted_order = sorted(range(len(parsed_dates)), key=lambda k: parsed_dates[k])
    # Map those indices back to actual date numbers (0..5)
    # Because date_cols is ['date0','date1',...], the index k corresponds to that number already.
    return sorted_order


def add_temporal_color_features(df: pd.DataFrame, sorted_order: list) -> pd.DataFrame:
    df = df.copy()

    # For each channel, reorder mean columns by chronological order and compute diffs/global change/volatility
    for c in ['red', 'green', 'blue']:
        mean_cols = [
            f'img_{c}_mean_date{i}'
            for i in sorted_order
            if f'img_{c}_mean_date{i}' in df.columns
        ]
        # If your dataset has date1..date5 (no date0), mean_cols may be shorter.
        if len(mean_cols) < 2:
            continue

        for j in range(len(mean_cols) - 1):
            df[f'img_{c}_mean_diff_{j}'] = df[mean_cols[j + 1]] - df[mean_cols[j]]

        df[f'img_{c}_mean_global_change'] = df[mean_cols[-1]] - df[mean_cols[0]]
        df[f'img_{c}_mean_std_across_time'] = df[mean_cols].std(axis=1)

    # NDVI-like index on most recent date available (chronological last)
    last_date = sorted_order[-1]
    red_col = f'img_red_mean_date{last_date}'
    green_col = f'img_green_mean_date{last_date}'
    if red_col in df.columns and green_col in df.columns:
        df['ndvi_like'] = (df[green_col] - df[red_col]) / (df[green_col] + df[red_col] + 1e-6)
    else:
        df['ndvi_like'] = np.nan

    return df


# Compute global chronological order once (assumes consistent mapping across rows)
sorted_order = compute_sorted_order_from_first_row(train_df)

# Apply feature engineering to BOTH train and test
train_df = add_temporal_color_features(train_df, sorted_order)
test_df  = add_temporal_color_features(test_df, sorted_order)

train_df = add_geometrical_features(train_df)
test_df  = add_geometrical_features(test_df)

# -------------------------
# 4) Feature lists
# -------------------------
geo_features = [
    'area', 'perimeter', 'compactness',
    'bbox_ratio', 'elongation',
    'convexity', 'n_vertices',
    'area_perimeter_ratio'
]

# All engineered temporal + raw img features start with img_
img_features = [c for c in train_df.columns if c.startswith('img_')]
# ndvi_like does not start with img_, so add explicitly
numeric_features = geo_features + img_features + ['ndvi_like']

categorical_features = ['urban_type', 'geography_type']
feature_cols = numeric_features + categorical_features

# Sanity: ensure features exist in both train/test
missing_train = [c for c in feature_cols if c not in train_df.columns]
missing_test  = [c for c in feature_cols if c not in test_df.columns]
if missing_train:
    raise KeyError(f"Missing in train_df: {missing_train[:20]} (and {len(missing_train)-20} more)" if len(missing_train)>20 else f"Missing in train_df: {missing_train}")
if missing_test:
    raise KeyError(f"Missing in test_df: {missing_test[:20]} (and {len(missing_test)-20} more)" if len(missing_test)>20 else f"Missing in test_df: {missing_test}")

# -------------------------
# 5) Preprocessor (fit on train only)
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), categorical_features),
    ],
    remainder='drop'
)

# -------------------------
# 6) Model factory
# -------------------------
def make_model():
    return xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=NUM_CLASSES,
        n_estimators=5000,        # large + early stopping in CV
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=1.5,
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

# -------------------------
# 7) Cross-validation
# -------------------------
y = train_df['change_type'].map(change_type_map).values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

f1s, accs, lls = [], [], []

print("\n======================")
print("CROSS-VALIDATION (5-fold)")
print("======================")

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), 1):
    tr_df = train_df.iloc[tr_idx]
    va_df = train_df.iloc[va_idx]
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    # Fit preprocessing on train fold only (no leakage)
    preproc_fold = clone(preprocessor)
    X_tr = preproc_fold.fit_transform(tr_df[feature_cols])
    X_va = preproc_fold.transform(va_df[feature_cols])

    w_tr = compute_sample_weight(class_weight='balanced', y=y_tr)

    model = make_model()
    model.set_params(early_stopping_rounds=100)

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )

    y_pred = model.predict(X_va)
    y_proba = model.predict_proba(X_va)

    f1 = f1_score(y_va, y_pred, average='macro')
    acc = accuracy_score(y_va, y_pred)
    ll = log_loss(y_va, y_proba, labels=np.arange(NUM_CLASSES))

    f1s.append(f1); accs.append(acc); lls.append(ll)
    print(f"Fold {fold}: macroF1={f1:.4f}  acc={acc:.4f}  logloss={ll:.4f}  best_iter={model.best_iteration}")

print("\nCV summary:")
print(f"macroF1: mean={np.mean(f1s):.4f} std={np.std(f1s):.4f}")
print(f"acc:     mean={np.mean(accs):.4f} std={np.std(accs):.4f}")
print(f"logloss: mean={np.mean(lls):.4f} std={np.std(lls):.4f}")

# -------------------------
# 8) Final training on full train + predict test
# -------------------------
print("\n======================")
print("FINAL TRAINING + SUBMISSION")
print("======================")

X_train_full = preprocessor.fit_transform(train_df[feature_cols])
y_train_full = y
w_full = compute_sample_weight(class_weight='balanced', y=y_train_full)

final_model = make_model()

# Optional: no early stopping here because no validation split;
# you can set n_estimators to int(np.mean([best iters]))+something if you want.
final_model.set_params(n_estimators=1000)

final_model.fit(X_train_full, y_train_full, sample_weight=w_full)

X_test = preprocessor.transform(test_df[feature_cols])
pred_int = final_model.predict(X_test)
pred_label = [inv_change_type_map[int(p)] for p in pred_int]

# Use test_df.index if it's the correct Id, otherwise adapt for Kaggle
submission = pd.DataFrame({
    "Id": test_df.index,
    "change_type": pred_label
})
submission.to_csv("xgb_submission.csv", index=False)

print("Submission saved to xgb_submission.csv")
print("Submission shape:", submission.shape)
print(submission.head())
