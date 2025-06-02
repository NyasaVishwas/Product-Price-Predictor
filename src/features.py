import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_features(df):
    df = df.copy()

    # Fill missing numerical features with median
    for col in ['retail_price', 'discounted_price']:
        df[col] = df[col].fillna(df[col].median())

    # Select features
    selected_features = ['retail_price', 'discounted_price', 'brand']

    # Handle missing categorical features with a placeholder
    df['brand'] = df['brand'].fillna('Unknown')

    label_encoders = {}
    for col in ['brand']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df[selected_features]

    return X, label_encoders
