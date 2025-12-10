# car_price_prediction_full.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for Windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# -------------------------------
# Load dataset
# -------------------------------
DATA_PATH = "data/car_data.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape} rows, {df.shape[1]} columns")
print(df.head())

# Detect target
TARGET_COL = "Selling_Price"
FEATURES = df.drop(columns=[TARGET_COL]).columns.tolist()

# Optional: create 'age' feature
df['age'] = 2025 - df['Year']  # assuming current year 2025
FEATURES.append('age')
FEATURES.remove('Year')

# -------------------------------
# Visualizations
# -------------------------------
def visualize(data, target_col):
    # Distribution of target
    plt.figure(figsize=(8,5))
    sns.histplot(data[target_col], kde=True)
    plt.title(f"{target_col} Distribution")
    plt.savefig("target_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10,8))
    numeric_cols = data.select_dtypes(include=['int64','float64']).columns
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()

    # Boxplots for categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure(figsize=(10,5))
        sns.boxplot(x=col, y=target_col, data=data)
        plt.title(f"{target_col} vs {col}")
        plt.savefig(f"{target_col}vs{col}.png")
        plt.close()

visualize(df, TARGET_COL)
print("Visualizations saved as PNG files.")

# -------------------------------
# Preprocessing
# -------------------------------
cat_features = df.select_dtypes(include=['object']).columns.tolist()
num_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
num_features.remove(TARGET_COL)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

# -------------------------------
# Train/Test Split
# -------------------------------
X = df[FEATURES]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# -------------------------------
# Build pipeline
# -------------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# -------------------------------
# Train model
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R^2 Score: {r2:.3f}")

# -------------------------------
# Save prediction results
# -------------------------------
results = X_test.copy()
results['Actual_Price'] = y_test
results['Predicted_Price'] = y_pred
results.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")