import geopandas as gpd
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ANALYSE COMPLÈTE DU DATASET")
print("="*70)

# Chargement des données
print("Chargement des données...")
train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)
print("Chargement terminé!")

print('\n### STRUCTURE GLOBALE ###')
print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')
print(f'Colonnes: {list(train_df.columns)}')

print('\n### DISTRIBUTION DE LA CIBLE ###')
target_dist = train_df['change_type'].value_counts()
print(target_dist)
print(f'\nRatio déséquilibre max/min: {target_dist.max() / target_dist.min():.2f}')

print('\n### FEATURES DISPONIBLES ###')
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != 'geometry']

print(f'Features numériques ({len(numeric_cols)}): {numeric_cols}')
print(f'Features catégorielles ({len(categorical_cols)}): {categorical_cols}')

print('\n### STATISTIQUES FEATURES NUMÉRIQUES ###')
print(train_df[numeric_cols].describe())

print('\n### DISTRIBUTION FEATURES CATÉGORIELLES ###')
for col in categorical_cols:
    print(f'\n{col}:')
    print(train_df[col].value_counts())

print('\n### VALEURS MANQUANTES ###')
print(train_df.isnull().sum())

print('\n### GÉOMÉTRIE ###')
train_df['area'] = train_df.geometry.area
print(f'Area min: {train_df["area"].min():.2f}')
print(f'Area max: {train_df["area"].max():.2f}')
print(f'Area median: {train_df["area"].median():.2f}')

print('\n### CORRÉLATIONS NUMÉRIQUES ###')
corr = train_df[numeric_cols + ['geometry']].select_dtypes(include=[np.number]).corr()
for col in numeric_cols:
    if col in corr.index:
        corr_vals = corr[col].sort_values(ascending=False)
        print(f'{col}: {corr_vals.to_dict()}')
