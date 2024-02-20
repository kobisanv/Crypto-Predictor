# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
loaded_model = tf.keras.models.load_model('bitcoin_price_predictor.h5')

# Define the cryptocurrency ticker symbol
ticker = 'BTC-USD'

# Download historical data for the specified cryptocurrency ticker symbol
btc_data = yf.download(ticker, period="max")

# Get user input for the future date they want to predict
user_input_date = input("Enter the future date you want to predict (YYYY-MM-DD): ")

# Convert the user input string to a datetime object
user_input_date = pd.to_datetime(user_input_date)

# Check if the user input date is in the future
if user_input_date <= pd.to_datetime('today'):
    print("Error: Please enter a future date.")
else:
    # Extract historical close prices up to the user input date
    historical_data = btc_data.loc[btc_data.index <= user_input_date]['Close'].values

    # Scale the historical data using MinMaxScaler
    scaler = MinMaxScaler()
    close_price = historical_data.reshape(-1, 1) # ensure range is within -1 to 1 
    close_fitscaled = scaler.fit_transform(close_price) # 

    # Define the sequence length for input data
    SEQ_LEN = 99

    # Prepare input sequences for prediction
    X_data = []
    for i in range(len(close_fitscaled) - SEQ_LEN): # subtracts to make sure enough data points left in close_fitscaled for creating sequences of length 99
        seq = close_fitscaled[i: i + SEQ_LEN] # sequence of 99 from index 
        X_data.append(seq) 
    X_data = np.array(X_data)

    # Make predictions using the loaded model
    predicted_prices_scaled = loaded_model.predict(X_data)

    # Inverse transform the scaled predictions to get actual prices
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

    # Print the predicted Bitcoin prices for the user-input future date
    print("Predicted Bitcoin prices for {}: {}".format(user_input_date.strftime('%Y-%m-%d'), predicted_prices[-1]))
