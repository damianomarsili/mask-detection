import cv2
from tensorflow import keras
import os
from face_detection import detect_faces, print_face, crop_image
import numpy as np

# Colors in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

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

def update_text(image, num_masks):
    font = cv2.FONT_HERSHEY_TRIPLEX
    thickness = 1
    font_scale = 1
    org = (0, 30)

    if num_masks > 0:
        image = cv2.putText(image, 'Detected ' + str(num_masks) + ' masks', org, font, font_scale, GREEN, thickness, cv2.LINE_AA)
    else:
        image = cv2.putText(image, 'No Masks Detected', org, font, font_scale, RED, thickness, cv2.LINE_AA)
    
    return image

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
        
        mask_counter = 0

        for face in faces:
            cropped_image = crop_image(image, face)
            detect_mask(image, model)
            
            if detect_mask(image, model):
                image = print_face(image, face, True) 
                mask_counter += 1
            else:
                image = print_face(image, face, False)

        image = update_text(image, mask_counter)
        cv2.imshow("Dami & Jocelyn's Mask Detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()



main()
