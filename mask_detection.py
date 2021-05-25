import cv2
from tensorflow import keras
from tensorflow.keras import layers
import os
from face_detection import detect_faces, print_face, crop_image

cap = cv2.VideoCapture(0)
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
        print_face(image, face, True)
        cropped_image = crop_image(image, face)
        cv2.imshow('Cropped image', cropped_image)


    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
