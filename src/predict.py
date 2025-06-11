import os
import sys
import argparse
import logging
import pandas as pd
import joblib
import numpy as np
from src.data_loader import load_data, basic_cleaning
from src.features import preprocess_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def predict(input_file, model_path='models/gradient_boosting.joblib', output_file='data/predictions.csv'):
    """
    Load new data, preprocess features, and make predictions using a trained model.
    """
    try:
        logging.info(f"🔍 Loading data from: {input_file}")
        df_new = load_data(input_file)
        df_new = basic_cleaning(df_new)

        logging.info("⚙️ Preprocessing features...")
        X_new, _ = preprocess_features(df_new, fit=False)

        logging.info(f"💾 Loading model from: {model_path}")
        model = joblib.load(model_path)

        logging.info("🔮 Making predictions...")
        y_pred_log = model.predict(X_new)
        y_pred = np.expm1(y_pred_log)  # inverse log1p if used during training

        # Save predictions
        df_new['predicted_price'] = y_pred
        df_new.to_csv(output_file, index=False)
        logging.info(f"✅ Predictions saved to {output_file}")

        # Print preview
        print(df_new[['product_name', 'predicted_price']].head())

    except Exception as e:
        logging.error(f"❌ An error occurred during prediction: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Product Price Predictor - Make predictions on new data.")
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input CSV file.'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/gradient_boosting.joblib',
        help='Path to the trained model file.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/predictions.csv',
        help='Path to save the predictions.'
    )
    args = parser.parse_args()

    predict(args.input, model_path=args.model, output_file=args.output)

if __name__ == "__main__":
    main()
