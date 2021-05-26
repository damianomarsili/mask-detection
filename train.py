import cv2 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory


def save_model(json_filename, weights_filename, model):
    # Save model as JSON
    model_json = model.to_json()
    with open(json_filename, 'w') as json_file:
        json_file.write(model_json)

    # Save weights as HDF5
    model.save_weights(weights_filename)


def train():
    pass


model = train()
save_model('mask_detetion.json', 'mask_detection.h5', model)
