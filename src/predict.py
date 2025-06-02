import pandas as pd
import joblib
from src.data_loader import load_data, basic_cleaning
from src.features import preprocess_features

def predict(filepath, model_path='models/linear_regression.joblib'):
    # Load new data
    df_new = load_data(filepath)
    df_new = basic_cleaning(df_new)

    # Preprocess features
    X_new, _ = preprocess_features(df_new)

    # Load model
    model = joblib.load(model_path)

    # Predict
    predictions = model.predict(X_new)

    # Attach predictions to dataframe
    df_new['predicted_price'] = predictions

    # Print and save predictions
    print(df_new[['product_name', 'predicted_price']].head())
    df_new.to_csv('data/predictions.csv', index=False)
    print("Predictions saved to data/predictions.csv")

if __name__ == "__main__":
    predict('data/raw_data.csv')
