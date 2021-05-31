import cv2
import os
import mediapipe as mp

# Color values in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

padding =  70

def detect_faces(image, height, width):
    """ Face detection using Mediapipe's face detection.

    Args:
        image (cv2): Input image to determine if faces are present.
        height (int): Height of input frame
        width (int): Width of input frame

    Returns:
        list: Tuple of faces detected (xmin, ymin, width, height)
    """
    mp_face_detection = mp.solutions.face_detection
    faces = []

    with mp_face_detection.FaceDetection(min_detection_confidence = 0.5) as face_detection:

        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections: 
                bounding_box = detection.location_data.relative_bounding_box
                # All values in detection normalized 0.0 -> 1.0
                face = [
                        int(bounding_box.xmin * width), 
                        int(bounding_box.ymin * height), 
                        int(bounding_box.width * width), 
                        int(bounding_box.height * height)
                        ]
                faces.append(face)
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
