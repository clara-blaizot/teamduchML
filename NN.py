# ======================
# IMPORTS
# ======================
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ======================
# CONFIGURATION
# ======================
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.3
NUM_CLASSES = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ======================
# LOAD GEOJSON DATA
# ======================
train_df = gpd.read_file("train.geojson", index_col=0)  

test_df  = gpd.read_file("test.geojson", index_col=0)

# ======================
# GEOMETRY FEATURES
# ======================
for df in [train_df, test_df]:
    df["area"] = df.geometry.area
    df["perimeter"] = df.geometry.length
    df["compactness"] = 4 * np.pi * df["area"] / (df["perimeter"] ** 2)

# ======================
# DATE FEATURES
# ======================
date_cols = ["date0", "date1", "date2", "date3", "date4"]
for df in [train_df, test_df]:
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df[date_cols] = df[date_cols].where(df[date_cols].notnull(), pd.NaT)
    dates_sorted = np.sort(df[date_cols].values, axis=1)
    for i in range(len(date_cols) - 1):
        delta = (pd.to_datetime(dates_sorted[:, i+1]) - pd.to_datetime(dates_sorted[:, i])).days
        df[f"delta_{i}_{i+1}"] = np.clip(delta, 0, None)
    df["total_duration"] = np.clip(
        (pd.to_datetime(dates_sorted[:, -1]) - pd.to_datetime(dates_sorted[:, 0])).days, 0, None
    )

# ======================
# TARGET
# ======================
change_type_map = {
    "Demolition": 0, "Road": 1, "Residential": 2,
    "Commercial": 3, "Industrial": 4, "Mega Projects": 5
}
y_train = train_df["change_type"].map(change_type_map)

# ======================
# FEATURES
# ======================
numeric_features = (
    ["area", "perimeter", "compactness", "total_duration"] +
    [c for c in train_df.columns if c.startswith("delta_")] +
    [c for c in train_df.columns if c.startswith("img_")]
)
categorical_features = ["urban_type", "geography_type"]

X_train = train_df[numeric_features + categorical_features]
X_test  = test_df[numeric_features + categorical_features]

# ======================
# PREPROCESSING
# ======================
# Fill missing numeric values
X_train[numeric_features] = X_train[numeric_features].fillna(X_train[numeric_features].median())
X_test[numeric_features]  = X_test[numeric_features].fillna(X_train[numeric_features].median())  # train median

# Encode categorical features safely for NN

label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()

    # Add explicit unknown token
    train_vals = X_train[col].astype(str).values
    le.fit(list(train_vals) + ["__unknown__"])
    label_encoders[col] = le

    # Transform train
    X_train[col] = le.transform(X_train[col].astype(str))

    # Transform test (map unseen → "__unknown__")
    X_test[col] = X_test[col].astype(str).apply(
        lambda x: x if x in le.classes_ else "__unknown__"
    )
    X_test[col] = le.transform(X_test[col])

# Standardize numeric features
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features]  = scaler.transform(X_test[numeric_features])

# ======================
# CUSTOM DATASET
# ======================
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.tensor(X_num.values, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat.values, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx]

X_train_num = X_train[numeric_features]
X_train_cat = X_train[categorical_features]
X_test_num  = X_test[numeric_features]
X_test_cat  = X_test[categorical_features]

train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================
# MODEL
# ======================
class TabularNN(nn.Module):
    def __init__(self, num_numeric, cat_dims, embedding_dims, hidden_dims, dropout=0.3, output_dim=6):
        super().__init__()
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, embedding_dims)
        ])
        emb_total = sum(embedding_dims)
        input_dim = num_numeric + emb_total
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(x_emb, dim=1)
        x = torch.cat([x_num, x_emb], dim=1)
        return self.network(x)

# Embedding sizes
cat_dims = [len(label_encoders[col].classes_) for col in categorical_features]


embedding_dims = [min(50, (size+2)//2) for size in cat_dims]

model = TabularNN(
    num_numeric=len(numeric_features),
    cat_dims=cat_dims,
    embedding_dims=embedding_dims,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
    output_dim=NUM_CLASSES
).to(DEVICE)

# ======================
# LOSS AND OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======================
# TRAINING LOOP
# ======================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for X_num_batch, X_cat_batch, y_batch in train_loader:
        X_num_batch = X_num_batch.to(DEVICE)
        X_cat_batch = X_cat_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_num_batch, X_cat_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_num_batch.size(0)
        
        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

### safety check

for i, col in enumerate(categorical_features):
    max_val = X_test_cat[col].max()
    if max_val >= cat_dims[i]:
        raise ValueError(
            f"Categorical feature '{col}' has value {max_val} "
            f"but embedding size is {cat_dims[i]}"
        )


# ======================
# PREDICTION
# ======================
model.eval()
with torch.no_grad():
    X_test_num_tensor = torch.tensor(X_test_num.values, dtype=torch.float32).to(DEVICE)
    X_test_cat_tensor = torch.tensor(X_test_cat.values, dtype=torch.long).to(DEVICE)
    preds = model(X_test_num_tensor, X_test_cat_tensor)
    test_labels = torch.argmax(preds, dim=1).cpu().numpy()

# ======================
# SAVE SUBMISSION
# ======================
submission = pd.DataFrame({
    "Id": range(len(test_labels)),
    "change_type": test_labels
})
submission.to_csv("nn_submission.csv", index=False)
print("Submission saved! Rows:", len(submission))
