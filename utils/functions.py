import numpy as np
from PIL import Image
import math
from deepface.commons import distance
from PIL import Image
def alignment_procedure_upd(img, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock
    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree
        if direction == -1:
            angle = 90 - angle
            img = Image.fromarray(img)
            img = np.array(img.rotate(direction * angle))

    # Resize the image to fit within the original boundaries
    height, width, _ = img.shape
    max_dim = max(height, width)
    img_resized = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    center_x, center_y = max_dim // 2, max_dim // 2
    img_resized[center_y - height // 2:center_y - height // 2 + height,
                center_x - width // 2:center_x - width // 2 + width] = img

    return img_resized
def alignment_procedure(img, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock
    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree
        if direction == -1:
            angle = 90 - angle
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    return img 
