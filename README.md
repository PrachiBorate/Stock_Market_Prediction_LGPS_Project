# Stock_Market_Prediction


This project provides a web-based system to Stock Market Prediction using various Python libraries generate trading signals based on historical stock data ,market indicators and web interfacing.

## Features 
- __Data Collection and Preprocessing__: Gather and clean historical stock market data, financial indicators, and news sentiment for analysis. 
- **Feature Engineering**: Create technical indicators, financial ratios, and sentiment scores from news data. 
- __Time Series Analysis__: Use ARIMA, SARIMA, or ETS models to forecast stock prices by analyzing trends, seasonality, and residuals. 
- __Machine Learning Models__: Implement algorithms like Linear Regression, Random Forest, and LSTM for stock trend prediction. 
- __Trading Signal Generation__: Develop algorithms to produce buy, hold, or sell signals based on predictions.
- __Backtesting__: Evaluate trading strategies on historical data to assess performance.
- __Web Application or Desktop Software__: Create an interface for visualizing stock predictions and trading signals.
- __Visualization__: Use Matplotlib, Seaborn, or Plotly to generate interactive charts and dashboards. 

## Technologies Used
- __Backend__: Python, Flask
- __Stock Market Prediction__: load_model, train_model, generate_signals, predict_future
- __Frontend__: HTML, Plotly
- __Deployment__: Flask

## Setup Instructions
### Prerequisites
- Python 3.x
- pip (Python package installer)                      


## Install Dependencies
pip install yfinance pandas numpy scikit-learn xgboost matplotlib flask joblib

## Web Application

### 1. Start the Flask App:

from flask import Flask, render_template, request

import pandas as pd

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly.io as pio

from model import train_model, generate_signals, predict_future, load_model

app = Flask(__name__)

#Load the model
model, scaler, features = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    #Load historical stock data (replace with your actual data loading logic)
    data_AAPL = pd.read_csv('data/AAPL.csv')
    data_GOOG = pd.read_csv('data/GOOG.csv')
    data_MSFT = pd.read_csv('data/MSFT.csv')
    
    #Assuming 'Date' column exists and is in datetime format
    data_AAPL['Date'] = pd.to_datetime(data_AAPL['Date'])
    data_GOOG['Date'] = pd.to_datetime(data_GOOG['Date'])
    data_MSFT['Date'] = pd.to_datetime(data_MSFT['Date'])
    
    #Set index to 'Date' column
    data_AAPL.set_index('Date', inplace=True)
    data_GOOG.set_index('Date', inplace=True)
    data_MSFT.set_index('Date', inplace=True)
    
    #Calculate four-year rolling average of volume for each company
    data_AAPL['Volume_4yr_avg'] = data_AAPL['Volume'].rolling(window='1460D').mean()
    data_GOOG['Volume_4yr_avg'] = data_GOOG['Volume'].rolling(window='1460D').mean()
    data_MSFT['Volume_4yr_avg'] = data_MSFT['Volume'].rolling(window='1460D').mean()
    
    #Prepare subplot for volume comparison and four-year average
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Volume Comparison', 'Four-Year Average Volume'))
    
    #Add trace for AAPL volume
    fig.add_trace(go.Scatter(x=data_AAPL.index, y=data_AAPL['Volume'], mode='lines', name='AAPL Volume', line=dict(color='blue')), row=1, col=1)
    
    #Add trace for GOOG volume
    fig.add_trace(go.Scatter(x=data_GOOG.index, y=data_GOOG['Volume'], mode='lines', name='GOOG Volume', line=dict(color='green')), row=1, col=1)
    
    #Add trace for MSFT volume
    fig.add_trace(go.Scatter(x=data_MSFT.index, y=data_MSFT['Volume'], mode='lines', name='MSFT Volume', line=dict(color='red')), row=1, col=1)
    
    #Add trace for AAPL four-year average volume
    fig.add_trace(go.Scatter(x=data_AAPL.index, y=data_AAPL['Volume_4yr_avg'], mode='lines', name='AAPL 4yr Avg Volume', line=dict(color='blue', dash='dash')), row=2, col=1)
    
    #Add trace for GOOG four-year average volume
    fig.add_trace(go.Scatter(x=data_GOOG.index, y=data_GOOG['Volume_4yr_avg'], mode='lines', name='GOOG 4yr Avg Volume', line=dict(color='green', dash='dash')), row=2, col=1)
    
    #Add trace for MSFT four-year average volume
    fig.add_trace(go.Scatter(x=data_MSFT.index, y=data_MSFT['Volume_4yr_avg'], mode='lines', name='MSFT 4yr Avg Volume', line=dict(color='red', dash='dash')), row=2, col=1)
    
    fig.update_layout(title_text='Volume Comparison and Four-Year Average', xaxis_title='Date', yaxis_title='Volume')
    graph_html = pio.to_html(fig, full_html=False)
    
    return render_template('predict.html', graph_html=graph_html)
if __name__ == '__main__':
    app.run(debug=True)

## 2. Running the Application:

- __Navigate to the Project Directory__: Open a terminal or command prompt and navigate to the directory where your Flask application (app.py) is located.

- __Execute the Command__: Run the command python app.py. This will start the Flask development server locally.

- __Access the Application__: Once the Flask server starts, open a web browser and go to "http://localhost:5000" or "http://127.0.0.1:5000" (Public URL). You should see your Flask application's homepage (index.html) rendered.

- __Interact with the Application__: Fill out any forms or navigate to different routes defined in your Flask application (/predict, etc.) to interact with the functionality.

- __Stopping the Application__: To stop the Flask server, go back to the terminal where it's running and press Ctrl + C. Confirm if prompted to terminate the server.

- This command (python app.py) is essential for starting a Flask application from the command line and is the standard way to run Flask apps during development. Adjust the script (run.py) and project structure as per your specific application requirements and environment.

