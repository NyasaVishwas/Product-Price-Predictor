import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_data, basic_cleaning
from src.features import preprocess_features
from src.model import evaluate_model, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    """
    Main pipeline for training the Product Price Predictor model.
    """
    try:
        logging.info("🔍 Loading and cleaning data...")
        df = load_data('data/raw_data.csv')
        df = basic_cleaning(df)

        logging.info("✅ Data loaded successfully.")

        # Separate features and target
        X = df.drop('discounted_price', axis=1)
        y = df['discounted_price']

        # Optional: log-transform the target
        y = np.log1p(y)  # comment out if not needed

        logging.info("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logging.info("⚙️ Preprocessing features...")
        X_train, label_encoders = preprocess_features(X_train, fit=True)
        X_test, _ = preprocess_features(X_test, label_encoders=label_encoders, fit=False)

        logging.info("🔍 Starting hyperparameter tuning...")
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

        best_model = grid_search.best_estimator_
        logging.info(f"✅ Best hyperparameters found: {grid_search.best_params_}")

        # Predict and inverse log transform
        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_orig = np.expm1(y_test)

        # Evaluate model
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        r2 = r2_score(y_test_orig, y_pred)
        logging.info(f"📈 RMSE after tuning: {rmse:.2f}")
        logging.info(f"📈 R² Score after tuning: {r2:.4f}")

        # Save model
        model_path = 'models/gradient_boosting.joblib'
        save_model(best_model, model_path)
        logging.info(f"💾 Model saved to {model_path}")

    except Exception as e:
        logging.error(f"❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
