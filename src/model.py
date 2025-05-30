from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model(X_train, y_train):
    """
    Train a linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print RMSE.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    print(f"Model RMSE: {rmse:.2f}")
    return rmse

def save_model(model, path='models/linear_regression.joblib'):
    """
    Save the trained model.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")
