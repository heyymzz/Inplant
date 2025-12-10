import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = {
    'Area': [800, 900, 1000, 1100, 1200, 1500],
    'Price': [40, 45, 50, 55, 60, 75]   
}
df = pd.DataFrame(data)
X = df[['Area']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Test Area values:\n", X_test)
print("\nActual Prices:\n", y_test)
print("\nPredicted Prices:\n", y_pred)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
