from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.merge import concatenate
from keras import backend as K
from keras.utils import plot_model


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

def deep_model(learning_rate=1e-5, state_size=10, num_of_actions=4):
    model = Sequential()
    model.add(Dense(512, input_shape=(state_size,)))
    model.add(PReLU())
    model.add(Dropout(0.2))

    for _ in range(2):
        model.add(Dense(512))
        model.add(PReLU())
        model.add(Dropout(0.2))

    model.add(Dense(num_of_actions))
    opt = Adam(learning_rate, epsilon=1e-8)
    model.compile(optimizer=opt, loss='mse')
    model.summary()
    return model

def dueldqn_model(learning_rate=1e-5, state_size=10, num_of_actions=4):

    inputS = Input(shape=(state_size,))

    value = Dense(state_size*3)(inputS)
    value = PReLU()(value)
    value = Dense(1, )(value)

    advantage = Dense(state_size*3)(inputS)
    advantage = PReLU()(advantage)
    advantage = Dense(num_of_actions, )(advantage)

    #No activation function before merge
    merge_layer = Lambda(dueldqn_formula, output_shape=(num_of_actions,))([value, advantage])
    model = Model(inputs=inputS, outputs=merge_layer)

    opt = Adam(learning_rate, epsilon=1e-8)
    model.compile(optimizer=opt, loss='mse')
    model.summary()
    # plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
    return model

def dueldqn_formula(x):
    return x[0]+(x[1] - K.mean(x[1]))

def conv2d_model(learning_rate=5e-5, state_shape=None, num_of_actions=4):
    batch, rows, cols, channels = state_shape
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', input_shape=(rows, cols, channels)))
    model.add(PReLU())
    model.add(Conv2D(filters=32, kernel_size=(2,2), padding='same'))
    model.add(PReLU())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same'))
    model.add(PReLU())
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same'))
    model.add(PReLU())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same'))
    model.add(PReLU())
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same'))
    model.add(PReLU())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same'))
    model.add(PReLU())
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same'))
    model.add(PReLU())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(num_of_actions))
    opt = Adam(lr=learning_rate, epsilon=1e-8)
    model.compile(optimizer=opt, loss='mse')
    model.summary()
    return model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    