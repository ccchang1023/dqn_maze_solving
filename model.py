from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU, PReLU


def default_model(learning_rate=1e-5, state_size=10, num_of_actions=4):
    model = Sequential()
    # Input shape should be: (batch, maze_size) 
    model.add(Dense(state_size*3, input_shape=(state_size,)))
    # model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))


    # model.add(Dense(state_size*2))
    # # model.add(BatchNormalization())
    # model.add(PReLU())
    # model.add(Dropout(0.2))

    model.add(Dense(num_of_actions))
    opt = Adam(learning_rate, epsilon=1e-8)
    # opt = RMSprop(learning_rate)
    model.compile(optimizer=opt, loss='mse')
    model.summary()
    return model
 
def conv2d_model(state_shape, num_of_actions):
    batch, rows, cols, channels = state_shape
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', input_shape=(rows, cols, channels)))
    model.add(PReLU())
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid'))
    model.add(PReLU())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid'))
    model.add(PReLU())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(PReLU())
    model.add(Dropout(0.1))
    model.add(Dense(num_of_actions))
    model.add(PReLU())
    
    opt = Adam(lr=1e-4, epsilon=1e-8)
    model.compile(optimizer=opt, loss='mse')
    return model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    