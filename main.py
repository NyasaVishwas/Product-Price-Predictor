import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, basic_cleaning
from src.model import train_model, evaluate_model, save_model
from src.features import preprocess_features

def main():
    # Load raw data
    df = load_data('data/raw_data.csv')
    
    # Basic cleaning
    df = basic_cleaning(df)
    
    # Preprocess features
    X, label_encoders = preprocess_features(df)
    y = df['price']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save
    save_model(model)

if __name__ == "__main__":
    main()