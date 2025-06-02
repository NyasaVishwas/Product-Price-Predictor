import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path='data/products.csv'):
    df = pd.read_csv(path)
    return df

def preprocess_features(df):
    """
    Select relevant features and handle missing values.
    """

    # Use 'retail_price' as target since 'price' is missing
    y = df['retail_price']
    X = df.drop(['retail_price', 'discounted_price', 'uniq_id', 'crawl_timestamp',
                 'product_url', 'pid', 'image', 'description', 'product_rating',
                 'overall_rating', 'product_specifications'], axis=1)

    # Handle missing values
    X = X.fillna('Unknown')
    y = y.fillna(y.median())  # or y.mean()

    # Encode 'product_category_tree' and 'brand'
    categorical_cols = ['product_category_tree', 'brand']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded_cats = pd.DataFrame(encoder.fit_transform(X[categorical_cols]),
                                columns=encoder.get_feature_names_out(categorical_cols))

    X = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True),
                   encoded_cats.reset_index(drop=True)], axis=1)

    # Optional: scale numeric features
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y
