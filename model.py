from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import ReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU, PReLU


def default_model(maze_size, num_of_actions):
    model = Sequential()
    # Input shape should be: (batch, maze_size) 
    model.add(Dense(maze_size, input_shape=(maze_size,)))
    model.add(PReLU())
    model.add(Dense(maze_size))
    model.add(PReLU())
    model.add(Dense(num_of_actions))
    model.compile(optimizer='adam', loss='mse')
    return model