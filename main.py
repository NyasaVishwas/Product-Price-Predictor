import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, basic_cleaning
from src.features import preprocess_features
from src.model import train_model, evaluate_model, save_model

def main():
    # Load and clean data
    df = load_data('data/raw_data.csv')
    df = basic_cleaning(df)

    # Preprocess features and target
    X, label_encoders = preprocess_features(df)
    y = df['discounted_price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

if __name__ == "__main__":
    main()
