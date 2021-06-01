import cv2 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from pathlib import Path


def save_model(json_filename, weights_filename, model):
    """ Save model and weights to JSON and HDF5.

    Args:
        json_filename (str): Name of JSON file.
        weights_filename (str): Name of weights file.
        model (keras.sequential): Trained neural network model.
    """
    # Save model as JSON
    model_json = model.to_json()
    with open(json_filename, 'w') as json_file:
        json_file.write(model_json)

    # Save weights as HDF5
    model.save_weights(weights_filename)

def load_png(specifier):
    """ Load and organize PNG files for training model.

    Args:
        specifier (str): Location of files.

    Returns:
        [2 numpy arr]: feature space and labels.
    """
    dirpath = 'data/' + specifier
    p = Path(dirpath)
    dirs = p.glob('*')
    labels = []
    data = []
    label_dict = {'mask' : 0, 'no-mask': 1}
    for folder in dirs:
        label = str(folder).split('/')[-1]
        print(label)
        for image_path in folder.glob('*.png'):
            img = image.load_img(image_path, target_size = (224, 224))
            data.append(image.img_to_array(img))
            labels.append(label_dict[label])
    image_train_png = np.array(data)
    label_train_png = np.array(labels)
    return image_train_png, label_train_png

def load_jpg(specifier):
    """  Load and organize JPG files for training model.

    Args:
        specifier (str): Location of files.

    Returns:
        [2 numpy arr]: feature space and labels.
    """
    dirpath = 'data/' + specifier
    p = Path(dirpath)
    dirs = p.glob('*')
    labels = []
    data = []
    label_dict = {'mask' : 0, 'no-mask': 1}
    for folder in dirs:
        label = str(folder).split('/')[-1]
        print(label)
        for image_path in folder.glob('*.jpg'):
            img = image.load_img(image_path, target_size = (224, 224))
            data.append(image.img_to_array(img))
            labels.append(label_dict[label])
    image_train_jpg = np.array(data)
    label_train_jpg = np.array(labels)
    return image_train_jpg, label_train_jpg

def load_jpeg(specifier):
    """  Load and organize JPEG files for training model.

    Args:
        specifier (str): Location of files.

    Returns:
        [2 numpy arr]: feature space and labels.
    """
    dirpath = 'data/' + specifier
    p = Path(dirpath)
    dirs = p.glob('*')
    labels = []
    data = []
    label_dict = {'mask' : 0, 'no-mask': 1}
    for folder in dirs:
        label = str(folder).split('/')[-1]
        print(label)
        for image_path in folder.glob('*.jpeg'):
            img = image.load_img(image_path, target_size = (224, 224))
            data.append(image.img_to_array(img))
            labels.append(label_dict[label])
    image_train_jpeg = np.array(data)
    label_train_jpeg = np.array(labels)
    return image_train_jpeg, label_train_jpeg


def load_ds_train():
    """ Load all training file types together and store in ds.

    Returns:
        [2 np arrays]: Collection of feature space and corresponding labels.
    """
    png_images, png_labels = load_png('train')
    jpg_images, jpg_labels = load_jpg('train')
    jpeg_images, jpeg_labels = load_jpeg('train')
    train_images = np.concatenate((png_images, jpg_images, jpeg_images))
    train_labels = np.concatenate((png_labels, jpg_labels, jpeg_labels))
    return train_images, train_labels

def load_ds_valid():
    """ Load all validation file types together and store in ds.

    Returns:
        [2 np arrays]: Collection of all validation images and corresponding labels.
    """
    png_images, png_labels = load_png('valid')
    jpg_images, jpg_labels = load_jpg('valid')
    jpeg_images, jpeg_labels = load_jpeg('valid')
    valid_images = np.concatenate((png_images, jpg_images, jpeg_images))
    valid_labels = np.concatenate((png_labels, jpg_labels, jpeg_labels))

    return valid_images, valid_labels

def load_ds_test():
    """ Load all test file types together and store in ds.

    Returns:
        [2 np arrays]: Collection of all test images and corresponding labels.
    """
    png_images, png_labels = load_png('test')
    jpg_images, jpg_labels = load_jpg('test')
    jpeg_images, jpeg_labels = load_jpeg('test')
    test_images = np.concatenate((png_images, jpg_images, jpeg_images))
    test_labels = np.concatenate((png_labels, jpg_labels, jpeg_labels))

    return test_images, test_labels

def preprocess_images(train_images, valid_images, test_images):
    """ Preprocess images by resizing and normalizing.

    Args:
        train_images (np array): Collection of all training images.
        valid_images (np array): Collection of all validation images.
        test_images (np array): Collection of all test images.

    Returns:
        [3 np arrays]: Processed images.
    """
    # Resize data appropriately
    size = train_images.shape[1]
    train_images = np.reshape(train_images, [-1, size, size, 3])
    valid_images = np.reshape(valid_images, [-1, size, size, 3])
    test_images = np.reshape(test_images, [-1, size, size, 3])

    # Normalize image
    train_image_mean = np.mean(train_images, axis = 0)
    train_images -= train_image_mean / 10
    valid_images -= train_image_mean / 10
    test_images -= train_image_mean / 10

    return train_images, valid_images, test_images

def preprocess_labels(train_labels, valid_labels, test_labels):
    """ Normalize labels into binary np array.

    Args:
        train_labels (np array): Collection of all training result labels.
        valid_labels (np array): Collection of all validation result labels.
        test_labels (np array): Collection of all testing result labels.

    Returns:
        [3 np arrays]: Binary labels.
    """
    num_labels = 2
    train_labels = to_categorical(train_labels)
    valid_labels = to_categorical(valid_labels)
    test_labels = to_categorical(test_labels)

    return train_labels, valid_labels, test_labels


def plot_metrics(history):
    """ Display training metrics.

    Args:
        history (np array): Validation results.
    """
    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()

# TODO: All choices below are entirely meaningless atm - need to experiment and optimize
def train(model):
    """ Train neural network model.

    Args:
        model (keras.Sequential): Trained model.
    """
    train_images, train_labels = load_ds_train()
    valid_images, valid_labels = load_ds_valid()
    test_images, test_labels = load_ds_test() 

    print('Train Data', train_images.shape, train_labels.shape)
    print('Validation Data', valid_images.shape, valid_labels.shape)
    print('Test Data', test_images.shape, test_labels.shape)

    train_images, valid_images, test_images = preprocess_images(train_images, valid_images, test_images)
    train_labels, valid_labels, test_labels = preprocess_labels(train_labels, valid_labels, test_labels)

    # early_stopping = callbacks.EarlyStopping(
        # min_delta = 0.005,
        # patience = 3, 
        # restore_best_weights = True)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 6e-5)
    model.compile(
            optimizer = optimizer,
            loss = 'binary_crossentropy',
            metrics = ['binary_accuracy'],
            )

    history = model.fit(
            train_images, 
            train_labels,
            batch_size = 10,
            epochs = 50, 
            validation_data = (valid_images, valid_labels))
            # callbacks = [early_stopping], 
            # verbose = 0)

    plot_metrics(history)
    

# TODO: All choices below are entriely meaningless atm - need to experiment and optimize
# TODO: Also need to check sizes/shapes I'm lost atm
def make_model():
    """ Set model parameters.

    Returns:
        keras.Sequential: Model with parameters and optimizers.
    """
    pretrained_base = tf.keras.applications.ResNet50(input_shape = (224, 224, 3)) # Need to check input_shape
    pretrained_base.trainable = False

    model = keras.Sequential([
        pretrained_base,
        layers.Flatten(),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(2, activation = 'softmax'),
        ])
    return model



model = make_model()
train(model)
save_model('mask_detection.json', 'mask_detection.h5', model)
