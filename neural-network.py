import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. MAPPING DES CLASSES ET CHARGEMENT DES DONNÉES
# ============================================================================
change_type_map = {
    'Demolition': 0, 
    'Road': 1, 
    'Residential': 2, 
    'Commercial': 3, 
    'Industrial': 4, 
    'Mega Projects': 5
}

# Lecture des données depuis les fichiers GeoJSON
train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

print(f"Shape données train: {train_df.shape}")
print(f"Shape données test: {test_df.shape}")

# ============================================================================
# 2. EXTRACTION DES FEATURES GÉOMÉTRIQUES
# ============================================================================
# Features basées sur la géométrie des polygones
train_df['area'] = train_df.geometry.area
train_df['perimeter'] = train_df.geometry.length
train_df['compactness'] = 4 * np.pi * train_df['area'] / (train_df['perimeter'] ** 2)

test_df['area'] = test_df.geometry.area
test_df['perimeter'] = test_df.geometry.length
test_df['compactness'] = 4 * np.pi * test_df['area'] / (test_df['perimeter'] ** 2)

# ============================================================================
# 3. SÉLECTION DES FEATURES
# ============================================================================
# Features numériques : géométrie + images (RGB moyennes/écarts-types sur 5 dates)
numeric_features = ['area', 'perimeter', 'compactness'] + \
                   [col for col in train_df.columns if col.startswith('img_')]

# Features catégorielles
categorical_features = ['urban_type', 'geography_type']

print(f"Features numériques: {numeric_features}")
print(f"Features catégorielles: {categorical_features}")

# ============================================================================
# 4. PIPELINE DE PREPROCESSING
# ============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Remplace NaN par médiane
            ('scaler', StandardScaler())  # Normalise entre -1 et 1
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Remplace NaN par le mode
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encode catégories
        ]), categorical_features)
    ])

# ============================================================================
# 5. PRÉPARATION DES DONNÉES
# ============================================================================
# Transformation des données d'entraînement
X_train = preprocessor.fit_transform(train_df[numeric_features + categorical_features])
y_train = train_df['change_type'].map(change_type_map).values

# Transformation des données de test
X_test = preprocessor.transform(test_df[numeric_features + categorical_features])

print(f"\nDimensions après preprocessing:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Split train/validation pour évaluer le modèle avant de soumettre
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"X_train_split shape: {X_train_split.shape}")
print(f"X_val shape: {X_val.shape}")

# ============================================================================
# 6. CONSTRUCTION DU RÉSEAU DE NEURONES
# ============================================================================
# Architecture : couches denses avec Dropout pour éviter l'overfitting

model = keras.Sequential([
    # Couche d'entrée
    layers.Input(shape=(X_train.shape[1],)),
    
    # Première couche cachée : 128 neurones avec activation ReLU
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),  # Normalise les activations (accélère apprentissage)
    layers.Dropout(0.3),           # Désactive 30% des neurones aléatoirement (évite overfitting)
    
    # Deuxième couche cachée : 64 neurones
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Troisième couche cachée : 32 neurones
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Couche de sortie : 6 neurones (une par classe) avec softmax
    layers.Dense(6, activation='softmax')
])

# Affiche le résumé de l'architecture
model.summary()

# ============================================================================
# 7. COMPILATION DU MODÈLE
# ============================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Optimiseur avec learning rate
    loss='sparse_categorical_crossentropy',                 # Loss pour classification multi-classe
    metrics=['accuracy']
)

# ============================================================================
# 8. ENTRAÎNEMENT DU MODÈLE
# ============================================================================
print("\nEntraînement du modèle...")

history = model.fit(
    X_train_split, y_train_split,
    epochs=100,                           # Nombre de passes sur l'ensemble d'entraînement
    batch_size=32,                        # Nombre d'échantillons traités avant mise à jour
    validation_data=(X_val, y_val),       # Données pour évaluation pendant l'entraînement
    verbose=1,
    callbacks=[
        # Early stopping : arrête si validation loss n'améliore pas depuis 15 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        # Réduit learning rate si plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
)

# ============================================================================
# 9. ÉVALUATION SUR LE VALIDATION SET
# ============================================================================
val_predictions = model.predict(X_val).argmax(axis=1)
val_accuracy = accuracy_score(y_val, val_predictions)

print(f"\nAccuracy sur validation set: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print("\nRapport de classification:")
print(classification_report(y_val, val_predictions, 
                          target_names=list(change_type_map.keys())))

# ============================================================================
# 10. PRÉDICTIONS SUR LE TEST SET
# ============================================================================
print("\nGénération des prédictions sur le test set...")

# Prédictions probabilistes (forme: (n_samples, 6))
y_test_probs = model.predict(X_test)

# Récupère la classe avec la probabilité maximale
y_test_pred = y_test_probs.argmax(axis=1)

print(f"Prédictions shape: {y_test_pred.shape}")
print(f"Classes prédites (unique): {np.unique(y_test_pred)}")

# ============================================================================
# 11. SAUVEGARDE DES RÉSULTATS
# ============================================================================
# Crée DataFrame avec les prédictions
pred_df = pd.DataFrame({
    'change_type': y_test_pred
})

# Sauvegarde en CSV pour la soumission
pred_df.to_csv("neural_network_submission.csv", index=True, index_label='Id')

print("\n✓ Fichier de soumission sauvegardé: neural_network_submission.csv")

# ============================================================================
# 12. AFFICHAGE DES STATISTIQUES
# ============================================================================
print("\n" + "="*60)
print("RÉSUMÉ DU MODÈLE")
print("="*60)
print(f"Architecture: 6 couches")
print(f"Total de paramètres: {model.count_params():,}")
print(f"Paramètres entraînables: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f"Epochs d'entraînement: {len(history.history['loss'])}")
print(f"Accuracy validation: {val_accuracy*100:.2f}%")
print("="*60)