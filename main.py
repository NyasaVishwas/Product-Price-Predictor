from src.data_loader import load_data, basic_cleaning
from src.features import encode_features
from src.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split
import joblib

def main():
    df = load_data("data/raw_data.csv")
    df = basic_cleaning(df)
    df, le_brand, le_cat = encode_features(df)
    
    X = df[['retail_price', 'product_rating', 'overall_rating', 'brand_encoded', 'category_encoded']]
    y = df['discounted_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    mae, r2 = evaluate_model(model, X_test, y_test)
    
    print(f"Model evaluation - MAE: {mae}, R2: {r2}")
    
    joblib.dump(model, "models/linear_regression.joblib")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
