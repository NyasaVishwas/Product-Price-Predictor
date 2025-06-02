import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    print(f"Model RMSE: {rmse:.2f}")
    print(f"Model R²: {r2:.2f}")

    # Residual Plot
    residuals = y_test - predictions
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=predictions, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('plots/residual_plot.png')
    plt.close()

    # Actual vs. Predicted
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=predictions)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    plt.savefig('plots/actual_vs_predicted.png')
    plt.close()

    print("Evaluation plots saved to 'plots/' directory.")

    return rmse, r2

def save_model(model, path='models/linear_regression.joblib'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")
