import pandas as pd
import joblib
from src.data_loader import load_data, basic_cleaning
from sklearn.preprocessing import LabelEncoder

def preprocess_features(df):
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['main_category', 'brand']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Select features
    feature_cols = ['product_rating', 'overall_rating', 'main_category', 'brand']
    X = df[feature_cols]

    return X, label_encoders

def predict(filepath, model_path='models/linear_regression.joblib'):
    df_new = load_data(filepath)
    df_new = basic_cleaning(df_new)

    X_new, _ = preprocess_features(df_new)

    model = joblib.load(model_path)
    predictions = model.predict(X_new)

    df_new['predicted_price'] = predictions

    print(df_new[['product_name', 'predicted_price']].head())
    df_new.to_csv('data/predictions.csv', index=False)
    print("Predictions saved to data/predictions.csv")

if __name__ == "__main__":
    predict('data/raw_data.csv')
