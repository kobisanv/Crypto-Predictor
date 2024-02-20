import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

ticker = 'DOGE-USD'
doge_data = yf.download(ticker, period="max")

scaler = MinMaxScaler()
close_price = doge_data.Close.values.reshape(-1, 1) # ensures in range of -1 to 1
close_fitscaled = scaler.fit_transform(close_price) 
close_fitscaled.shape # prints out the rows and columns
np.isnan(close_fitscaled).any() # handles NaNs
close_fitscaled = close_fitscaled[~np.isnan(close_fitscaled)] # removes NaNs
close_fitscaled = close_fitscaled.reshape(-1, 1) # ensures still in range of -1 to 1 after removing NaNs
np.isnan(close_fitscaled.any()) # removes any NaNs

SEQ_LEN = 100 # determines how much each group of input output pairs

def to_seqeuences(data, seq_len):
    d = []
    
    for i in range(len(data) - seq_len): # deletes 100 from length of dataset for every iteration
        d.append(data[i: i + seq_len]) # create a sequence of 100 starting from i 
    
    return np.array(d) 

def preprocess(data_raw, seq_len, train_split): # train_split determines proportion of training to test data
    data = to_seqeuences(data_raw, seq_len) # return a sequence of length 100
    num_train = int(train_split * data.shape[0]) # data.shape[0] reps total # of sequences
    
    # input sequences for training
    X_train = data[:num_train, :-1, :] # from start to num_train index excluding last element from each sequence, as it is target value
    # corresponding target values for training sequences above
    y_train = data[:num_train, -1, :] # includes last elements from each sequence for the training set
    
    # input sequences for training
    X_test = data[num_train, :-1, :] # from num train index to end exclusind last element from each sequence
    # corresponding target values for training sequences above
    y_test = data[num_train, -1, :] # includes last elements from each sequence for testing
    
    return X_train, y_train, X_test, y_test
 
X_train, y_train, X_test, y_test = preprocess(close_fitscaled, SEQ_LEN, train_split = 0.90)
# 90 percent of data is training, 10 percent of data is testing

X_train.shape
X_test.shape

DROPOUT = 0.2 # 20 percent of neurons are dropped out to prevent overfitting 
# overfitting is when model performs well on training data but fails to perform as well with new data
WINDOW_SIZE = SEQ_LEN - 1 # number of time steps in each X sequence

# LSTM Model
# Bidirectional - learns from past and future time steps (learns in forward and reverse simultaneously)
# Build the LSTM model
# Two LSTM models used to capture short-term patterns and then long-term patterns 
model = tf.keras.Sequential()  # Create a sequential model

# Add a Bidirectional LSTM layer with return sequences (short-term patterns)
# Input shape: (WINDOW_SIZE, X_train.shape[-1]), specifying the number of time steps and features
model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1]))) # window_size -> # of time steps, X_train.shape[-1] -> # of features

# Add a Dropout layer to prevent overfitting
# Dropout randomly sets a fraction of input units to 0 during training, in this case 20%
model.add(Dropout(rate=DROPOUT))

# Add another Bidirectional LSTM layer with doubled units and return sequences (feeds more data for long-term patterns)
model.add(Bidirectional(LSTM(WINDOW_SIZE * 2, return_sequences=True)))

# Add a Dropout layer after the second Bidirectional LSTM layer
model.add(Dropout(rate=DROPOUT))

# Add a Bidirectional LSTM layer without returning sequences
model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

# Add a Dense layer with a single output unit for regression
model.add(Dense(units=1))

# Apply linear activation to the output
# Linear activation is used for regression problems to predict continuous numerical values
model.add(Activation('linear'))

# Compile the model
model.compile(
    loss='mean_squared_error',  # Use mean squared error as the loss function for regression
    optimizer='adam'  # Use the Adam optimizer for optimization
)

# Train the model
# Training the model with the specified parameters, including number of epochs, batch size, and validation split
history = model.fit(
    X_train, # input
    y_train, # output
    epochs=10, # 10 iterations
    batch_size=64, # each epoch uses 64 training examples
    shuffle=False,  # Do not shuffle the training data before each epoch
    validation_split=0.1  # Use 10% of the training data for validation during training
)

model.save('dogecoin_price_predictor.h5')