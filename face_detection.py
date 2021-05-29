import cv2
import os

# Color values in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

padding =  70

# Helper function to load face-detection cascade file
def load_cascade():
    face_cascade = cv2.CascadeClassifier()
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    if not face_cascade.load(cv2.samples.findFile(haar_model)):
        print('--(!)Error loading face cascade')
        exit(0)

    return face_cascade

# Face detection using OpenCV's detect cascade detection, returns list of faces detected
def detect_faces(image):
    face_cascade = load_cascade()
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)

    faces = face_cascade.detectMultiScale(image_gray)
    
    return faces

# Returns an image cropped by the face bounds plus padding
def crop_image(image, face):
    x, y, w, h = face
    crop_img = image[y - 112 : y + 112, x - 112: x + 112]
    return crop_img

# Prints face with color corresponding to if face is masked or not.
def print_face(image, face, is_masked):
    x, y, w, h = face
    if is_masked:
        image = cv2.rectangle(image, (x - padding // 2, y - padding // 2), (x + w + padding // 2, y + h + padding // 2), GREEN)
    else:
        image = cv2.rectangle(image, (x - padding // 2, y - padding // 2), (x + w + padding // 2, y + h + padding // 2), RED)

    return image
