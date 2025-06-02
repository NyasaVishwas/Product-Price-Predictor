import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def basic_cleaning(df):
    # Convert ratings to numeric
    df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
    df['overall_rating'] = pd.to_numeric(df['overall_rating'], errors='coerce')
    df['product_rating'] = df['product_rating'].fillna(df['product_rating'].mean())
    df['overall_rating'] = df['overall_rating'].fillna(df['overall_rating'].mean())
    df['brand'] = df['brand'].fillna("Unknown")

    # Extract main_category
    df['main_category'] = df['product_category_tree'].apply(
        lambda x: x.split('>>')[0].strip() if pd.notnull(x) else "Unknown"
    )
    
    # Drop rows with missing prices (target variable)
    df = df.dropna(subset=['retail_price', 'discounted_price'])

    return df
