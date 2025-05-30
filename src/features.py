import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_features(df):
    """
    Feature engineering: encode categorical variables and handle missing values
    """
    # Encode categorical variables
    label_encoders = {}
    for col in ['brand', 'main_category', 'sub_category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Select relevant features
    feature_cols = [
        'product_rating',
        'overall_rating',
        'brand',
        'main_category',
        'sub_category'
    ]

    X = df[feature_cols].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    return X_imputed, label_encoders  # optionally return encoders
