import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Model RMSE: {rmse:.2f}")

    # Plot Residuals
    residuals = y_test - predictions
    plt.figure(figsize=(8, 5))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

    # Plot Actual vs. Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs. Predicted Price')
    plt.show()

def save_model(model, filename='models/linear_regression.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")