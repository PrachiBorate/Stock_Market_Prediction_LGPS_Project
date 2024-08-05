# Import neccessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np
import pickle
import os

def load_model():
    model_path = 'models/model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        scaler = None   
        features = None  
    else:
        # Create a placeholder model if the file does not exist
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor(strategy='mean')
        model.fit(np.zeros((10, 1)), np.zeros(10))
        scaler = None
        features = None
    return model, scaler, features

def train_model(data):
    # Feature Engineering: Add Moving Averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Drop rows with NaN values generated from rolling windows
    data = data.dropna()

    # Split the data into features and target
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200']]
    y = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train the RandomForest model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train_imputed, y_train)

    # Predictions
    predictions = model.predict(X_test_imputed)
    
    return predictions, y_test

def generate_signals(data, predictions):
    if 'Close' not in data.columns:
        raise KeyError("'Close' column is missing in the data")

    data['Prediction'] = pd.Series(predictions, index=data.index[-len(predictions):])
    data['Signal'] = 0
    data['Signal'][data['Prediction'] > data['Close'].shift(1)] = 1  # Buy signal
    data['Signal'][data['Prediction'] < data['Close'].shift(1)] = -1 # Sell signal
    return data

def predict_future(model, scaler, features, start_date, end_date):
    # Placeholder implementation
    dates = pd.date_range(start=start_date, end=end_date)
    predictions = model.predict(np.zeros((len(dates), 1))) 
    return predictions, dates
