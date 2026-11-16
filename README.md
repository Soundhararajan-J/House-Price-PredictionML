# ============================================================
# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# Algorithm: Random Forest Regression
# Dataset: house_price_dataset.csv
# ============================================================

# STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

sns.set_theme(style="whitegrid")

print("Libraries Imported Successfully!")


# STEP 2: LOAD DATASET

df = pd.read_csv("house_price_dataset.csv")

print("\nDATASET LOADED SUCCESSFULLY")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())


# STEP 3: DATASET INFORMATION

print("\nDATASET INFORMATION")

df.info()

print("\nSTATISTICAL SUMMARY")

print(df.describe())


# STEP 4: CHECK MISSING VALUES

print("\nMISSING VALUES")

print(df.isnull().sum())

print("\nDUPLICATE RECORDS:", df.duplicated().sum())


# STEP 5: DATA CLEANING

df = df.drop_duplicates().copy()

df["Price_INR"] = pd.to_numeric(
    df["Price_INR"], errors="coerce"
)

df = df.dropna(subset=["Price_INR"])

# Fill missing numerical values
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:
    if col not in ["House_ID", "Price_INR"]:
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values
categorical_columns = df.select_dtypes(
    include="object"
).columns

for col in categorical_columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nDATA CLEANING COMPLETED")

print("Remaining Missing Values:", df.isnull().sum().sum())


# STEP 6: EXPLORATORY DATA ANALYSIS

# A. House Price Distribution

plt.figure(figsize=(8, 5))

plt.hist(
    df["Price_INR"] / 100000,
    bins=30,
    edgecolor="black"
)

plt.title("House Price Distribution")
plt.xlabel("House Price (Lakhs)")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.show()


# B. Area vs House Price

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Area_sqft"],
    df["Price_INR"],
    alpha=0.6
)

plt.title("Area vs House Price")
plt.xlabel("Area (Square Feet)")
plt.ylabel("House Price (INR)")

plt.tight_layout()
plt.show()


# C. Average House Price by City

city_price = df.groupby(
    "City"
)["Price_INR"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))

city_price.plot(kind="bar")

plt.title("Average House Price by City")
plt.xlabel("City")
plt.ylabel("Average Price (INR)")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# D. Correlation Heatmap

plt.figure(figsize=(10, 6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()


# STEP 7: FEATURE SELECTION

X = df.drop(
    columns=[
        "House_ID",
        "Price_INR",
        "Price_Category"
    ],
    errors="ignore"
)

y = df["Price_INR"]

print("\nINPUT FEATURES:")
print(X.columns.tolist())

print("\nTARGET VARIABLE: Price_INR")


# STEP 8: TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING RECORDS:", X_train.shape[0])
print("TESTING RECORDS:", X_test.shape[0])


# STEP 9: DATA PREPROCESSING

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nDATA PREPROCESSING COMPLETED")


# STEP 10: RANDOM FOREST MODEL

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

print("\nRANDOM FOREST MODEL TRAINED SUCCESSFULLY!")


# STEP 11: PREDICTION

y_pred = model.predict(X_test)

print("\nHOUSE PRICE PREDICTION COMPLETED")

print("\nFIRST 10 PREDICTED PRICES:")

for price in y_pred[:10]:
    print("₹", round(price, 2))


# STEP 12: MODEL EVALUATION

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error:", round(mae, 2))

print("Mean Squared Error:", round(mse, 2))

print("Root Mean Squared Error:", round(rmse, 2))

print("R2 Score:", round(r2, 4))

print("R2 Percentage:", round(r2 * 100, 2), "%")


# STEP 13: ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot(
    [low, high],
    [low, high],
    "r--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")

plt.legend()
plt.tight_layout()
plt.show()


# STEP 14: COMPARE ACTUAL AND PREDICTED PRICES

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results["Difference"] = (
    results["Actual Price"] -
    results["Predicted Price"]
).abs()

print("\nACTUAL VS PREDICTED RESULTS")

print(results.head(10).round(2))


# STEP 15: NEW HOUSE PRICE PREDICTION

new_house = pd.DataFrame([{
    "City": "Chennai",
    "Property_Type": "Apartment",
    "Area_sqft": 1500,
    "BHK": 3,
    "Bathrooms": 2,
    "Age_years": 5,
    "Floor": 4,
    "Parking_Slots": 1,
    "Distance_to_City_Center_km": 8.0,
    "Furnishing": "Semi-Furnished",
    "Facing": "East",
    "Lift": "Yes",
    "Gym": "No",
    "Swimming_Pool": "No"
}])

predicted_price = model.predict(new_house)[0]

print("\n========== HOUSE PRICE PREDICTION ==========")

print("City: Chennai")
print("Property Type: Apartment")
print("Area: 1500 Sqft")
print("BHK: 3")

print("Predicted House Price: ₹",
      round(predicted_price, 2))

print("Price in Lakhs: ₹",
      round(predicted_price / 100000, 2),
      "Lakhs")


# STEP 16: SAVE PREDICTION RESULTS

results.to_csv(
    "house_price_predictions.csv",
    index=False
)

print("\nPrediction Results Saved Successfully!")

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")
