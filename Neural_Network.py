import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_neural_network(network_structure):
    # Initialize the neural network model
    model = Sequential()

    # Add each layer to the model
    for i, neurons in enumerate(network_structure):
        if i == 0:
            # Add the input layer
            model.add(Dense(neurons, activation='relu', input_shape=(input_shape,)))
        else:
            # Add hidden layers
            model.add(Dense(neurons, activation='relu'))

    # Add the output layer
    model.add(Dense(output_neurons, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model