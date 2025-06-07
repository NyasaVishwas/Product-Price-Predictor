import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    """Train Random Forest Regressor model."""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,        # try different depths
        min_samples_split=5  # minimum samples required to split
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Model RMSE: {rmse:.2f}")

    # Plot residuals
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()

    # Plot actual vs. predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs. Predicted Price')
    plt.show()

def save_model(model, model_path='models/random_forest_regressor.joblib'):
    """Save the trained model."""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
