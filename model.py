from keras.models import model_from_json

def load_model():
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("models/model.h5")
    return model