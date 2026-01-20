
# Imports
import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import joblib

# Paths
os.makedirs("model", exist_ok=True)

SCALER_PATH = "ml/models/scaler.pkl"
FEATURES_PATH = "ml/models/feature_cols.pkl"
MODEL_PATH = "ml/models/lstm_stock_model.h5"

# Load Dataset
df = pd.read_parquet("data/clean_market.parquet")

df = df.rename(columns={
    "date": "Date",
    "ticker": "Company"
})

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Company", "Date"]).reset_index(drop=True)

print("Dataset Shape:", df.shape)
print("Companies:", df["Company"].nunique())

# 🔥 Add Lagged Features (BIGGEST ACCURACY BOOST)
for lag in [1, 2, 3, 5]:
    df[f"close_lag_{lag}"] = df.groupby("Company")["close"].shift(lag)
    df[f"return_lag_{lag}"] = df.groupby("Company")["close"].pct_change(lag)

df = df.dropna().reset_index(drop=True)

# Feature Selection
feature_cols = [
    col for col in df.columns
    if col not in ["Date", "Company", "close"]
]

print("Total Features:", len(feature_cols))

# Scaling
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

joblib.dump(scaler, SCALER_PATH)
joblib.dump(feature_cols, FEATURES_PATH)

print("✅ Scaler & features saved")

# Sequence Creation
sequence_length = 20
threshold = 0.015  # 🔥 stricter threshold = cleaner labels

X, y = [], []

for company in df["Company"].unique():
    company_df = df[df["Company"] == company].reset_index(drop=True)

    for i in range(sequence_length, len(company_df)):
        X.append(
            company_df.iloc[i-sequence_length:i][feature_cols].values
        )

        prev_close = company_df["close"].iloc[i-1]
        next_close = company_df["close"].iloc[i]
        pct_change = (next_close - prev_close) / prev_close

        if pct_change > threshold:
            y.append(2)   # BUY
        elif pct_change < -threshold:
            y.append(0)   # SELL
        else:
            y.append(1)   # HOLD

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Class Weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Train / Val / Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False
)

# 🔥 Improved Model (Less Overfitting)
model = Sequential([
    Conv1D(64, 3, activation="relu",
           input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(2),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0007),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5
)

# Training
model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# Save Model
model.save(MODEL_PATH)
print("✅ Model saved")

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["SELL", "HOLD", "BUY"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 🔥 Confidence-Based Prediction (IMPORTANT)
label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

confidence_threshold = 0.60

sample_probs = model.predict(X_test[:1])[0]
confidence = np.max(sample_probs)
label = np.argmax(sample_probs)

if confidence < confidence_threshold:
    final_label = "HOLD"
else:
    final_label = label_map[label]

print("\n✅ Sample Prediction:", final_label,
      f"Confidence: {confidence*100:.2f}%")
