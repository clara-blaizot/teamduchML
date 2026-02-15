import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Chargement silent des données
train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

# Sauvegarde résumé dans un fichier
with open('dataset_analysis.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("ANALYSE COMPLÈTE DU DATASET\n")
    f.write("="*70 + "\n\n")
    
    f.write(f'Train shape: {train_df.shape}\n')
    f.write(f'Test shape: {test_df.shape}\n')
    f.write(f'Colonnes: {list(train_df.columns)}\n\n')
    
    f.write("### DISTRIBUTION DE LA CIBLE ###\n")
    target_dist = train_df['change_type'].value_counts()
    f.write(str(target_dist) + "\n")
    f.write(f'Total classes: {len(target_dist)}\n')
    f.write(f'Ratio déséquilibre max/min: {target_dist.max() / target_dist.min():.2f}\n\n')
    
    f.write("### FEATURES DISPONIBLES ###\n")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != 'geometry']
    
    f.write(f'Features numériques ({len(numeric_cols)}): {numeric_cols}\n')
    f.write(f'Features catégorielles ({len(categorical_cols)}): {categorical_cols}\n\n')
    
    f.write("### PREMIÈRES LIGNES ###\n")
    f.write(str(train_df.head()) + "\n\n")
    
    f.write("### STATISTIQUES FEATURES NUMÉRIQUES ###\n")
    f.write(str(train_df[numeric_cols].describe()) + "\n\n")
    
    f.write("### DISTRIBUTION FEATURES CATÉGORIELLES ###\n")
    for col in categorical_cols:
        f.write(f'\n{col}:\n')
        f.write(str(train_df[col].value_counts()) + "\n")
    
    f.write("\n### VALEURS MANQUANTES ###\n")
    f.write(str(train_df.isnull().sum()) + "\n\n")
    
    f.write("### GÉOMÉTRIE ###\n")
    train_df['area'] = train_df.geometry.area
    f.write(f'Area min: {train_df["area"].min():.2f}\n')
    f.write(f'Area max: {train_df["area"].max():.2f}\n')
    f.write(f'Area median: {train_df["area"].median():.2f}\n')

print("Analyse sauvegardée dans dataset_analysis.txt")
