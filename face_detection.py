import cv2
import os

# Color values in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

padding =  70
 
def load_cascade():
    """ Helper function to load face-detection cascade file.

    Returns:
        cv2: face-detection cascade file.
    """
    face_cascade = cv2.CascadeClassifier()
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    if not face_cascade.load(cv2.samples.findFile(haar_model)):
        print('--(!)Error loading face cascade')
        exit(0)

    return face_cascade

def detect_faces(image):
    """ Face detection using OpenCV's detect cascade detection.

    Args:
        image (cv2): Input image to determine if faces are present.

    Returns:
        list: All faces detected.
    """
    face_cascade = load_cascade()
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)

    faces = face_cascade.detectMultiScale(image_gray)
    
    return faces

def crop_image(image, face):
    """ Crop image in according to the face bounds plus additional padding.

    Args:
        image (int[]): Array of image pixels.
        face (tuple): Bounds of face.

    Returns:
        int[]: Cropped image.
    """
    x, y, w, h = face
    crop_img = image[y - padding // 2: y + h + padding // 2, x - padding // 2: x + h + padding // 2]
    return crop_img
    

def print_face(image, face, is_masked):
    """ Prints face with color corresponding to if face is masked or not.

    Args:
        image (int[]): Array of image pixels.
        face (tuple): Bounds of face.
        is_masked (boolean): True if mask, false otherwise.

    Returns:
        int[]: Colored image of face with or without mask.
    """
    x, y, w, h = face
    if is_masked:
        image = cv2.rectangle(image, (x - padding // 2, y - padding // 2), (x + w + padding // 2, y + h + padding // 2), GREEN)
    else:
        image = cv2.rectangle(image, (x - padding // 2, y - padding // 2), (x + w + padding // 2, y + h + padding // 2), RED)

    return image
