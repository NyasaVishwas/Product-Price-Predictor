import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_features(df):
    """Preprocess features for modeling."""
    # Select features (you can customize these)
    feature_cols = ['product_name', 'retail_price', 'brand', 'product_category_tree']

    # Fill missing values in numerical columns
    df['retail_price'] = df['retail_price'].fillna(df['retail_price'].median())

    # Encode categorical columns
    label_encoders = {}
    for col in ['product_name', 'brand', 'product_category_tree']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # Ensure string type
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df[feature_cols]
    return X, label_encoders
