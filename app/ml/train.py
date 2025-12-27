import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

print("🚀 Starting model training...")

# 1. Load the data
print("\n📂 Loading data from CSV...")
df = pd.read_csv('data/sample_data.csv')
print(f"✅ Loaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")

# 2. Feature engineering
print("\n🔧 Preprocessing features...")

# Categorical columns that need encoding
categorical_cols = ['vehicle_type', 'road_condition', 'weather_condition']

# Create label encoders for categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   Encoded {col}: {list(le.classes_)}")

# Select features for the model
feature_columns = [
    'distance_km',
    'traffic_hours',
    'vehicle_avg_speed',
    'vehicle_type_encoded',
    'road_condition_encoded',
    'weather_condition_encoded'
]

X = df[feature_columns]
y = df['actual_time_minutes']

print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# 3. Split data into training and testing sets
print("\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# 4. Train XGBoost model
print("\n🤖 Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=200,        # Number of boosting rounds
    max_depth=6,             # Maximum tree depth
    learning_rate=0.1,       # Step size shrinkage
    subsample=0.8,           # Fraction of samples for each tree
    colsample_bytree=0.8,    # Fraction of features for each tree
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

model.fit(X_train, y_train, verbose=False)
print("✅ Model trained successfully!")

# 5. Evaluate the model
print("\n📊 Evaluating model performance...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Training metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

# Testing metrics
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("\n📈 Training Set Performance:")
print(f"   MAE:  {train_mae:.2f} minutes")
print(f"   RMSE: {train_rmse:.2f} minutes")
print(f"   R²:   {train_r2:.4f}")

print("\n📈 Testing Set Performance:")
print(f"   MAE:  {test_mae:.2f} minutes")
print(f"   RMSE: {test_rmse:.2f} minutes")
print(f"   R²:   {test_r2:.4f}")

# Show some example predictions
print("\n🔍 Sample predictions vs actual:")
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred_test[idx]
    print(f"   Actual: {actual:.2f} min | Predicted: {predicted:.2f} min | Error: {abs(actual-predicted):.2f} min")

# 6. Save the model and encoders
print("\n💾 Saving model and encoders...")
os.makedirs('models', exist_ok=True)

# Save model
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✅ Model saved to models/xgboost_model.pkl")

# Save label encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("   ✅ Encoders saved to models/label_encoders.pkl")

# Save feature names for later use
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("   ✅ Feature names saved to models/feature_names.pkl")

print("\n🎉 Training complete! Model is ready for predictions.")
