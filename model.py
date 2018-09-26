from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.activations import relu
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU, PReLU


def restore_model(path):
    return load_model(path)

def default_model(state_size, num_of_actions):
    model = Sequential()
    # Input shape should be: (batch, maze_size) 
    model.add(Dense(state_size, input_shape=(state_size,)))
    model.add(PReLU())
    model.add(Dense(state_size))
    model.add(PReLU())
    model.add(Dense(num_of_actions))
    opt = Adam(lr=1e-4, epsilon=1e-8)
    model.compile(optimizer=opt, loss='mse')
    return model