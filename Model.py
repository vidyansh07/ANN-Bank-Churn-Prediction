import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_dim):
    """
    Builds and compiles a simple Artificial Neural Network.
    Uses 'sigmoid' activation for all layers as requested.
    """
    # Initializes the ANN
    ann = Sequential()
    
    # Adding the input layer and the first hidden layer
    ann.add(Dense(units=6, activation='sigmoid', input_dim=input_dim))
    
    # Adding the second hidden layer
    ann.add(Dense(units=6, activation='sigmoid'))
    
    # Adding the output layer
    ann.add(Dense(units=1, activation='sigmoid'))
    
    # Compiling the ANN
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return ann
