from keras.models import clone_model


model = None
targetModel = None

def set_model(input):
    global model
    model = input

def init_targetModel():
    global model
    global targetModel
    targetModel = clone_model(model)

def get_model():
    global model
    return model

def get_targetModel():
    global targetModel
    return targetModel

def update_targetModel():
    global model
    global targetModel
    # print("Cloning model...")
    targetModel.set_weights(model.get_weights())