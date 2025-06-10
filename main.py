import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from src.data_loader import load_data, basic_cleaning
from src.features import preprocess_features
from src.model import evaluate_model, save_model

def main():
    # Load and clean data
    df = load_data('data/raw_data.csv')
    df = basic_cleaning(df)

    # Separate features and target
    X = df.drop('discounted_price', axis=1)
    y = df['discounted_price']

    # Optional: log transform the target if skewed
    y = np.log1p(y)  # comment out if not needed

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess features
    X_train, label_encoders = preprocess_features(X_train, fit=True)
    X_test, _ = preprocess_features(X_test, label_encoders=label_encoders, fit=False)

    # Set up hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    gbr = GradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=gbr,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters:")
    print(grid_search.best_params_)

    # Use best model
    best_model = grid_search.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)
    y_pred = np.expm1(y_pred)  # Inverse log1p transform

    # Evaluate model performance
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))
    r2 = r2_score(np.expm1(y_test), y_pred)
    print(f'RMSE after tuning: {rmse:.2f}')
    print(f'R² Score after tuning: {r2:.4f}')

    # Save the model
    save_model(best_model, model_path='models/gradient_boosting.joblib')

if __name__ == "__main__":
    main()
