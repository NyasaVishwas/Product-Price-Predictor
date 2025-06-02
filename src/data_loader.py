import pandas as pd

def load_data(filepath):
    """
    Loads raw data from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def basic_cleaning(df):
    """
    Performs basic data cleaning:
    - Removes duplicates
    - Trims whitespace
    - Handles missing values if necessary
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Trim whitespace from object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Drop rows where 'retail_price' is missing (since it's our target)
    df = df.dropna(subset=['retail_price'])

    # Fill remaining missing values with 'Unknown' or suitable default
    df = df.fillna('Unknown')

    return df
