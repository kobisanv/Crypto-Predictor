import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# ---------------------------
# 1. Data Download & Preprocessing
# ---------------------------
ticker = 'ETH-USD'
eth_data = yf.download(ticker, period="max")

# Use all quantitative features: Open, High, Low, Close, Volume
quant_data = eth_data[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Scale all features between 0 and 1
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(quant_data)

# Remove any rows with NaN values
scaled_features = scaled_features[~np.isnan(scaled_features).any(axis=1)]

SEQ_LEN = 100  # Total length of each sequence (including target)

def to_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i: i + seq_len])
    return np.array(sequences)

def preprocess(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    # For each sequence, use the first seq_len-1 timesteps as input and the last timestep's closing price (index 3) as target.
    X_train = data[:num_train, :-1, :]  # shape: (num_train, seq_len-1, num_features)
    y_train = data[:num_train, -1, 3]     # index 3 corresponds to 'Close'
    
    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, 3]
    
    # Reshape targets to (samples, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, y_train, X_test, y_test

# Split data: 90% for training, 10% for testing
X_train, y_train, X_test, y_test = preprocess(scaled_features, SEQ_LEN, train_split=0.90)

DROPOUT = 0.2      # 20% dropout to prevent overfitting
WINDOW_SIZE = SEQ_LEN - 1  # Number of timesteps in each input sequence

# ---------------------------
# 2. Build the LSTM Model (Using all quantitative features)
# ---------------------------
model = tf.keras.Sequential()

# First Bidirectional LSTM layer (captures short-term patterns)
model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))

# Second Bidirectional LSTM layer with more units (captures longer-term patterns)
model.add(Bidirectional(LSTM(WINDOW_SIZE * 2, return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

# Third Bidirectional LSTM layer without returning sequences
model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

# Dense layer to produce a single output (predicting the closing price)
model.add(Dense(units=1))
model.add(Activation('linear'))

# Compile the model with Mean Squared Error loss and the Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# ---------------------------
# 3. Train the Model
# ---------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    shuffle=False,
    validation_split=0.1
)

# Save the trained model
model.save('ethereum_price_predictor.h5')
