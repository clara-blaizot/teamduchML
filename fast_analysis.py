import json
import pandas as pd
from collections import Counter

print("Lecture du fichier train.geojson...")

# Lecture JSON brute plus rapide
with open('train.geojson', 'r') as f:
    data = json.load(f)

features = data['features']
print(f"Total d'enregistrements: {len(features)}")

# Extraction des properties
properties_list = [f['properties'] for f in features]
df = pd.DataFrame(properties_list)

print(f"\nShape: {df.shape}")
print(f"Colonnes: {list(df.columns)}")

print("\n### DISTRIBUTION CIBLE ###")
target_counts = Counter([f['properties']['change_type'] for f in features])
for cls, count in sorted(target_counts.items(), key=lambda x: -x[1]):
    print(f"{cls}: {count}")

print(f"\n### TYPES DE DONNÉES ###")
print(df.dtypes)

print(f"\n### PREMIÈRES LIGNES ###")
print(df.head())

print(f"\n### STATISQUES NUMÉRIQUES ###")
numeric_df = df.select_dtypes(include=['number'])
print(numeric_df.describe())

# Sauvegarde pour consultation
with open('data_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write(f"Total samples: {len(features)}\n")
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Colonnes: {list(df.columns)}\n\n")
    f.write("### DISTRIBUTION CIBLE ###\n")
    for cls, count in sorted(target_counts.items(), key=lambda x: -x[1]):
        f.write(f"{cls}: {count}\n")
    f.write(f"\n{df.dtypes}\n\n")
    f.write(str(df.head()) + "\n\n")
    f.write(str(numeric_df.describe()) + "\n")

print("\nRésumé sauvegardé dans data_summary.txt")
