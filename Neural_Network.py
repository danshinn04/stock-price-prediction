# Neural_Network.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Preprocess the data to create sequences
def create_sequences(data, timesteps=60):
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))
X_train, y_train = create_sequences(scaled_data)

# Reshape X_train for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Create the LSTM model if no user input defined?
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create the LSTM model given user defined network_structure
def create_dynamic_lstm_model(input_shape, network_structure, dropout_rate=0.2):
    """
    Creates an LSTM model based on the network structure provided by the user.
    
    Args:
    - input_shape: Tuple representing the input shape, e.g., (timesteps, features)
    - network_structure: List of integers where each integer represents the number of neurons in that layer
    - dropout_rate: Float representing the dropout rate for regularization
    
    Returns:
    - A compiled Keras Sequential model
    """
    model = Sequential()
    
    # Add the first LSTM layer
    model.add(LSTM(units=network_structure[0], return_sequences=True if len(network_structure) > 1 else False, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Add any additional LSTM layers
    for layer_size in network_structure[1:-1]:  # Exclude the first and the last layer
        model.add(LSTM(units=layer_size, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    # Add the final LSTM layer without return_sequences
    if len(network_structure) > 1:
        model.add(LSTM(units=network_structure[-1]))
        model.add(Dropout(dropout_rate))
    
    # Add the output layer
    model.add(Dense(units=1))  # Assuming a single value prediction, e.g., price
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Train the model
def train_model(model, X, y, epochs=100, batch_size=32):
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Predict the price using LSTM
def predict_price(model, data, scaler):
    # Assuming `data` is the last `timesteps` days of prices
    last_sequence = scaler.transform(data.reshape(-1, 1))
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
    predicted_price = model.predict(last_sequence)
    # Inverse transform to get the actual price
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price
