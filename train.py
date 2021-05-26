import cv2 
import pandas as pd
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

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

# TODO: See hidden code in first lesson on Kaggle (Or setup block on exercise of first lesson)
def load_ds_train():
    pass

# TODO: See hidden code in first lesson on Kaggle (Or setup block on exercise of first lesson)
def load_ds_valid():
    pass

def plot_metrics(history):
   history_frame = pd.DataFrame(history.history)
   history_frame.loc[:, ['loss', 'val_loss']].plot()
   history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()

# TODO: All choices below are entirely meaningless atm - need to experiment and optimize
def train(model):
    ds_train = load_ds_train()
    ds_valid = load_ds_valid()
    
    optimizer = tf.keras.optimizers.Adam(epsilon = 0.01)
    model.compile(
            optimizer = optimizer,
            loss = 'binary_crossentropy',
            metrics = ['binary_accuracy'],
            )

    history = model.fit(
            ds_train,
            validation_data = ds_valid,
            epochs = 30,
            )

    plot_metrics(history)

# TODO: All choices below are entriely meaningless atm - need to experiment and optimize
# TODO: Also need to check sizes/shapes I'm lost atm
def make_model():
    pretrained_base = tf.keras.applications.ResNet50() # Need to check if need to change input_shape param
    pretrained_base.trainable = False

    model = keras.Sequential([
        pretrained_base,
        layers.Flatten(),
        layers.Dense(15, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid'),
        ])
    return model



model = make_model()
train(model)
save_model('mask_detetion.json', 'mask_detection.h5', model)
