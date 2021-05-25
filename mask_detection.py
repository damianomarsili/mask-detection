import cv2
from tensorflow import keras
from tensorflow.keras import layers
import os
from face_detection import detect_faces, print_face, crop_image

def load_model(json, weights):
    # Load json and create model
    json_file = open(json, 'r')
    model_json = json_file.read()
    json_file.close()

    model = keras.models.model_from_json(model_json)

    # Load weights into model
    model.load_weights(weights)
    return model

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
            #print_face(image, face, True)
            cropped_image = crop_image(image, face)
            
            # TODO: send cropped_image to model to evaluate 

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()



main()
