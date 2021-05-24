import cv2
import os

def load_cascade():
    face_cascade = cv2.CascadeClassifier()
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    if not face_cascade.load(cv2.samples.findFile(haar_model)):
        print('--(!)Error loading face cascade')
        exit(0)

    return face_cascade

def detect_face(image):
    face_cascade = load_cascade()
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)

    faces = face_cascade.detectMultiScale(image_gray)
    
    # Temporarily print
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        image = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (255, 0, 0), 4)
        cv2.imshow('Dami Jocely face detection', image)

#def print_face(face):

