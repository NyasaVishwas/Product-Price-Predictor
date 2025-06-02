import pandas as pd

def load_data(filepath):
    """Load data from CSV file."""
    df = pd.read_csv(filepath)
    return df

def basic_cleaning(df):
    """Basic cleaning: handle missing values and drop unnecessary columns."""
    # Drop rows where target variable (discounted_price) is missing
    df = df.dropna(subset=['discounted_price'])
    return df
