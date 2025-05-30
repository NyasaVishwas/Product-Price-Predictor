import joblib

def load_model(filepath):
    """
    Load a trained model from file.
    """
    return joblib.load(filepath)

def load_label_encoders():
    le_brand = joblib.load("models/le_brand.joblib")
    le_cat = joblib.load("models/le_cat.joblib")
    return le_brand, le_cat