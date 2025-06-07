import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_features(df, label_encoders=None, fit=True):
    """Preprocess features for modeling."""
    feature_cols = ['retail_price', 'brand', 'product_category_tree']

    # Fill missing values
    df['retail_price'] = df['retail_price'].fillna(df['retail_price'].median())
    for col in ['brand', 'product_category_tree']:
        df[col] = df[col].astype(str).fillna('Unknown')

    if fit:
        label_encoders = {}
        for col in ['brand', 'product_category_tree']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    else:
        for col in ['brand', 'product_category_tree']:
            le = label_encoders[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    X = df[feature_cols]
    return X, label_encoders
