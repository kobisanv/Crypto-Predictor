import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

ticker = 'SOL-USD'
# Download historical data for Solana
sol_data = yf.download(ticker, period="max")

# Use multiple columns: Open, High, Low, Close, Volume
quant_data = sol_data[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Scale the features to a range (typically 0 to 1)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(quant_data)

# Remove any rows containing NaN values (if any)
scaled_features = scaled_features[~np.isnan(scaled_features).any(axis=1)]
# At this point, scaled_features has shape (num_samples, 5)

SEQ_LEN = 100  # Total length of each sequence (including target)

def to_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i: i + seq_len])
    return np.array(sequences)

def preprocess(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    # For inputs, take the first seq_len-1 time steps (all features)
    X_train = data[:num_train, :-1, :]  # shape: (num_train, seq_len-1, num_features)
    # For targets, we use the closing price from the last time step (index 3)
    y_train = data[:num_train, -1, 3]     # shape: (num_train,)
    
    # For testing, use the remainder of the sequences
    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, 3]
    
    # Reshape targets to be 2D (samples, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test

# Split the data: 90% for training, 10% for testing
X_train, y_train, X_test, y_test = preprocess(scaled_features, SEQ_LEN, train_split=0.90)

DROPOUT = 0.2  # Fraction of neurons to drop for regularization
WINDOW_SIZE = SEQ_LEN - 1  # Number of time steps in each input sequence

# Build the LSTM model using Bidirectional LSTMs
model = tf.keras.Sequential()
# First Bidirectional LSTM layer with return_sequences=True for short-term patterns
model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))
# Second Bidirectional LSTM layer with more units
model.add(Bidirectional(LSTM(WINDOW_SIZE * 2, return_sequences=True)))
model.add(Dropout(rate=DROPOUT))
# Third Bidirectional LSTM layer without returning sequences
model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))
# Dense layer to produce a single output (predicting the closing price)
model.add(Dense(units=1))
model.add(Activation('linear'))

# Compile the model with mean squared error loss and the Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model (adjust epochs and batch size as needed)
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    shuffle=False,
    validation_split=0.1
)

# Save the trained model
model.save('solana_price_predictor.h5')
