from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from model import train_model, generate_signals, predict_future, load_model

# Initializing Flask App
app = Flask(__name__)

# Load the model
model, scaler, features = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    # Load historical stock data 
    data_AAPL = pd.read_csv('data/AAPL.csv')
    data_GOOG = pd.read_csv('data/GOOG.csv')
    data_MSFT = pd.read_csv('data/MSFT.csv')
    
    # Assuming 'Date' column exists and is in datetime format
    data_AAPL['Date'] = pd.to_datetime(data_AAPL['Date'])
    data_GOOG['Date'] = pd.to_datetime(data_GOOG['Date'])
    data_MSFT['Date'] = pd.to_datetime(data_MSFT['Date'])
    
    # Set index to 'Date' column
    data_AAPL.set_index('Date', inplace=True)
    data_GOOG.set_index('Date', inplace=True)
    data_MSFT.set_index('Date', inplace=True)
    
    # Calculate four-year rolling average of volume for each company
    data_AAPL['Volume_4yr_avg'] = data_AAPL['Volume'].rolling(window='1460D').mean()
    data_GOOG['Volume_4yr_avg'] = data_GOOG['Volume'].rolling(window='1460D').mean()
    data_MSFT['Volume_4yr_avg'] = data_MSFT['Volume'].rolling(window='1460D').mean()
    
    # Prepare subplot for volume comparison and four-year average
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Volume Comparison', 'Four-Year Average Volume'))
    
     # Add trace for AAPL volume
    fig.add_trace(go.Scatter(x=data_AAPL.index, y=data_AAPL['Volume'], mode='lines', name='AAPL Volume', line=dict(color='blue')), row=1, col=1)
    
    # Add trace for GOOG volume
    fig.add_trace(go.Scatter(x=data_GOOG.index, y=data_GOOG['Volume'], mode='lines', name='GOOG Volume', line=dict(color='green')), row=1, col=1)
    
    # Add trace for MSFT volume
    fig.add_trace(go.Scatter(x=data_MSFT.index, y=data_MSFT['Volume'], mode='lines', name='MSFT Volume', line=dict(color='red')), row=1, col=1)
    
    # Add trace for AAPL four-year average volume
    fig.add_trace(go.Scatter(x=data_AAPL.index, y=data_AAPL['Volume_4yr_avg'], mode='lines', name='AAPL 4yr Avg Volume', line=dict(color='blue', dash='dash')), row=2, col=1)
    
    # Add trace for GOOG four-year average volume
    fig.add_trace(go.Scatter(x=data_GOOG.index, y=data_GOOG['Volume_4yr_avg'], mode='lines', name='GOOG 4yr Avg Volume', line=dict(color='green', dash='dash')), row=2, col=1)
    
    # Add trace for MSFT four-year average volume
    fig.add_trace(go.Scatter(x=data_MSFT.index, y=data_MSFT['Volume_4yr_avg'], mode='lines', name='MSFT 4yr Avg Volume', line=dict(color='red', dash='dash')), row=2, col=1)
    
    fig.update_layout(title_text='Volume Comparison and Four-Year Average', xaxis_title='Date', yaxis_title='Volume')
    graph_html = pio.to_html(fig, full_html=False)
    
    return render_template('predict.html', graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)