#Install required libraries
%pip install numpy 
%pip install pandas 
%pip install matplotlib 
%pip install seaborn 
%pip install scikit-learn

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

sns.set_theme(style="whitegrid")

#Load dataset & Dataset information
df = pd.read_csv("house_price_dataset.csv")

print("Dataset Loaded Successfully")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

display(df.head())

print("Dataset Information")

df.info()

print("\nStatistical Summary")

display(df.describe())

#Check missing values
print("Missing Values in Dataset")

print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

#Data cleaning
# Remove duplicate records
df = df.drop_duplicates().copy()

# Convert target column to numeric
df["Price_INR"] = pd.to_numeric(
    df["Price_INR"], errors="coerce"
)

# Remove missing target values
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

print("Data Cleaning Completed Successfully!")

print("Remaining Missing Values:")
print(df.isnull().sum().sum())

#7. Exploratory Data Analysis (EDA)
#A. House price distribution
plt.figure(figsize=(8,5))

plt.hist(
    df["Price_INR"] / 100000,
    bins=30,
    edgecolor="black"
)

plt.title("House Price Distribution")
plt.xlabel("House Price (Lakhs)")
plt.ylabel("Number of Houses")

plt.show()

#B. Area vs house price
plt.figure(figsize=(8,5))

plt.scatter(
    df["Area_sqft"],
    df["Price_INR"],
    alpha=0.6
)

plt.title("Area vs House Price")
plt.xlabel("Area (Square Feet)")
plt.ylabel("House Price (INR)")

plt.show()

#C. Average house price by city
city_price = df.groupby(
    "City"
)["Price_INR"].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))

city_price.plot(kind="bar")

plt.title("Average House Price by City")
plt.xlabel("City")
plt.ylabel("Average Price (INR)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#D. Correlation heatmap
plt.figure(figsize=(10,6))

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

#Feature selection
# Remove unnecessary columns
X = df.drop(
    columns=[
        "House_ID",
        "Price_INR",
        "Price_Category"
    ],
    errors="ignore"
)

# Target variable
y = df["Price_INR"]

print("Input Features:")
print(X.columns.tolist())

print("\nTarget Variable: Price_INR")

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])

#Data preprocessing
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

print("Data Preprocessing Completed!")

#Build Random Forest model
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

print("Random Forest Model Trained Successfully!")

#Predict house prices
y_pred = model.predict(X_test)

print("House Price Prediction Completed!")

print("\nFirst 10 Predicted Prices:")

for price in y_pred[:10]:
    print("₹", round(price, 2))

#Model evaluation
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error:", round(mae, 2))

print("Mean Squared Error:", round(mse, 2))

print("Root Mean Squared Error:", round(rmse, 2))

print("R2 Score:", round(r2, 4))

print("R2 Percentage:", round(r2 * 100, 2), "%")

#Actual vs predicted graph
plt.figure(figsize=(8,6))

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
    "r--"
)

plt.title("Actual vs Predicted House Prices")

plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")

plt.show()

#Compare actual and predicted prices
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results["Difference"] = (
    results["Actual Price"] -
    results["Predicted Price"]
)

display(results.head(10).round(2))

#Predict a new house price
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

predicted_price = model.predict(new_house)

print("========== HOUSE PRICE PREDICTION ==========")

print("City: Chennai")

print("Property Type: Apartment")

print("Area: 1500 Sqft")

print("BHK: 3")

print("Predicted House Price: ₹",
      round(predicted_price[0], 2))

print("Price in Lakhs: ₹",
      round(predicted_price[0] / 100000, 2),
      "Lakhs")

#Save prediction results
results.to_csv(
    "house_price_predictions.csv",
    index=False
)

print("Prediction Results Saved Successfully!")
