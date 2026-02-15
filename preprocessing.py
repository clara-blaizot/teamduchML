import geopandas as gpd  
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  
import pandas as pd
import xgboost as xgb

# Mapping des classes cibles (change_type) vers des entiers pour la classification
change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5}

# Lecture des données d'entraînement et de test depuis les fichiers GeoJSON
train_df = gpd.read_file('train.geojson', index_col=0) 
test_df = gpd.read_file('test.geojson', index_col=0)

# Extraction de features géométriques avancées
train_df['area'] = train_df.geometry.area 
train_df['perimeter'] = train_df.geometry.length 
train_df['compactness'] = 4 * np.pi * train_df['area'] / (train_df['perimeter'] ** 2) 

# Même chose pour le test set
test_df['area'] = test_df.geometry.area
test_df['perimeter'] = test_df.geometry.length
test_df['compactness'] = 4 * np.pi * test_df['area'] / (test_df['perimeter'] ** 2)


# Sélection des features numériques : géométrie + statistiques d'images (moyennes et écarts-types RGB pour 5 dates)
numeric_features = ['area', 'perimeter', 'compactness'] + \
                   [col for col in train_df.columns if col.startswith('img_')] 

# Sélection des features catégorielles
categorical_features = ['urban_type', 'geography_type'] 

# Définition du preprocessing avec imputation :
# Pour numériques : imputer avec médiane puis normaliser
# Pour catégorielles : imputer avec valeur la plus fréquente puis one-hot encoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Remplace NaN par médiane
            ('scaler', StandardScaler())  # Puis normalise
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Remplace NaN par valeur la plus fréquente
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Puis encode
        ]), categorical_features)
    ])


# Préparation des données d'entraînement
X_train = preprocessor.fit_transform(train_df[numeric_features + categorical_features])  
y_train = train_df['change_type'].map(change_type_map)  

# Préparation des données de test 
X_test = preprocessor.transform(test_df[numeric_features + categorical_features])

# Affichage des dimensions pour vérifier
print(f"Dimensions entraînement: X={X_train.shape}, y={y_train.shape}")
print(f"Dimensions test: X={X_test.shape}")
