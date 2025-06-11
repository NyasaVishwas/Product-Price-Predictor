# 🛍️ Product Price Predictor

Predict product prices using machine learning based on product details like name, category, brand, and other features. The model is built using Gradient Boosting Regressor with hyperparameter tuning.

## 🚀 Features

- Data cleaning and preprocessing
- Feature engineering with consistent label encoding
- Hyperparameter tuning with GridSearchCV
- Model evaluation with RMSE and R²
- Save and load trained models
- Predict new product prices using saved models
- Logging and error handling for transparency
- CLI support for both training and prediction

## 📂 Project Structure

product-price-predictor/
    ├── data/
    │   ├── raw_data.csv
    │   └── predictions.csv
    ├── models/
    │   ├── gradient_boosting.joblib
    │   └── label_encoders.joblib
    ├── src/
    │   ├── data_loader.py
    │   ├── features.py
    │   ├── model.py
    │   └── predict.py
    ├── main.py
    ├── requirements.txt
    └── README.md

## 🔧 Installation

1️⃣ Clone the repository:
    git clone https://github.com/NyasaVishwas/Product-Price-Predictor.git
    cd Product-Price-Predictor

2️⃣ Create and activate a virtual environment:
    python3 -m venv venv
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate.bat  # Windows

3️⃣ Install dependencies:
    pip install -r requirements.txt

## 🏃‍♀️ Usage

1️⃣ Train the Model:
    python3 main.py
    - Trains the Gradient Boosting model with hyperparameter tuning.
    - Saves the model and label encoders to 'models/'.

2️⃣ Predict New Data:
    python3 -m src.predict --input data/raw_data.csv --model models/gradient_boosting.joblib --encoder models/label_encoders.joblib --output data/predictions.csv
    - Uses saved model and label encoders to predict prices for new data.
    - Outputs predictions to 'data/predictions.csv'.

## 📊 Results

Example prediction output:

| product_name                           | predicted_price |
|----------------------------------------|-----------------|
| Alisha Solid Women's Cycling Shorts    | 429.76          |
| FabHomeDecor Fabric Double Sofa Bed    | 21838.01        |
| AW Bellies                             | 488.30          |

## 🧩 Requirements

See 'requirements.txt' for the full list.

## 📈 Future Work

- Deploy model as a REST API using Flask/FastAPI
- Add interactive dashboards for feature importance and residuals
- Integrate unit tests with pytest

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

