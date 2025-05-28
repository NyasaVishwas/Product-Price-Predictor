from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    le_brand = LabelEncoder()
    le_cat = LabelEncoder()
    
    df['brand_encoded'] = le_brand.fit_transform(df['brand'])
    df['category_encoded'] = le_cat.fit_transform(df['main_category'])
    
    return df, le_brand, le_cat
