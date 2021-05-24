import cv2
from tensorflow import keras
from tensorflow.keras import layers
import os
from face_detection import detect_face

cap = cv2.VideoCapture(0)
run = True

while run:
    success, image = cap.read()
    
    if not success:
        print('Empty camera frame ignored')
        continue

    # flip image along y axis for selfie-view
    image = cv2.flip(image, 1)
    detect_face(image)    

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
