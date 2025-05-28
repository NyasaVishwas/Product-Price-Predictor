import joblib

def load_model(filepath):
    return joblib.load(filepath)

def predict_price(model, X_new):
    return model.predict(X_new)
