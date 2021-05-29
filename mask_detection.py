import cv2
from tensorflow import keras
import os
from face_detection import detect_faces, print_face, crop_image
import numpy as np

# Loads model from files
def load_model(json, weights):
    # Load json and create model
    json_file = open(json, 'r')
    model_json = json_file.read()
    json_file.close()

    model = keras.models.model_from_json(model_json)

    # Load weights into model
    model.load_weights(weights)
    return model

# Applies image smoothing
def blur_image(image, image_size):
    image = cv2.blur(image, (5,5))
    image = cv2.resize(image, image_size)
    return image

def detect_mask(image, model):
    x = []
    image_size = (224, 224)
    image = blur_image(image, image_size)
    x.append(image)
    image = np.array(x)
    return model.predict(image)[0][0] > 0.5


def main():
    cap = cv2.VideoCapture(0)
    model = load_model('mask_detection.json', 'mask_detection.h5')
    run = True

    while run:
        success, image = cap.read()
    
        if not success:
            print('Empty camera frame ignored')
            continue

        # flip image along y axis for selfie-view
        image = cv2.flip(image, 1)
        faces = detect_faces(image)    
    
        for face in faces:
            cropped_image = crop_image(image, face)
            detect_mask(image, model)
            
            if detect_mask(image, model):
                print_face(image, face, True) 
            else:
                print_face(image, face, False)
            

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()



main()
