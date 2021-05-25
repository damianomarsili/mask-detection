import cv2 
from tensorflow import keras
from tensorflow.keras import layers


def save_model(json_filename, weights_filename, model):
    # Save model as JSON
    model_json = model.to_json()
    with open(json_filename, 'w') as json_file:
        json_file.write(model_json)

    # Save weights as HDF5
    model.save_weights(weights_filename)


