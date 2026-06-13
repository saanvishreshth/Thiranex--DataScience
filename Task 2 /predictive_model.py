import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("dataset.csv")

print("Dataset Preview:")
print(df.head())

# Features and Target
X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Save Model
joblib.dump(model, "prediction_model.pkl")

print("\nModel saved as prediction_model.pkl")

# Test Prediction
new_house = [[2800, 5, 4]]

predicted_price = model.predict(new_house)

print("\nPredicted House Price:")
print(predicted_price[0])
