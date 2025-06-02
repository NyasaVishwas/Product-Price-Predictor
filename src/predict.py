import pandas as pd
import joblib
from src.features import preprocess_features

def predict(new_data_path, model_path='models/linear_regression.joblib'):
    # Load the saved model
    model = joblib.load(model_path)

    # Load new data
    df_new = pd.read_csv(new_data_path)

    # Basic cleaning and preprocessing
    X_new, label_encoders = preprocess_features(df_new)

    # Predict
    predictions = model.predict(X_new)

    # Add predictions to the dataframe
    df_new['predicted_price'] = predictions

    # Save results
    df_new.to_csv('data/predicted_results.csv', index=False)
    print("Predictions saved to data/predicted_results.csv")
