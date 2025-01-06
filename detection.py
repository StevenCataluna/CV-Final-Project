import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


#https://www.youtube.com/watch?v=d59wPaenvzs
#https://www.youtube.com/watch?v=B76GR7v_YMU
# https://www.youtube.com/watch?v=WAAjCOKIxwM

# Load in model
model = models.load_model('models/traffic_sign_detector6.h5')

# Load in test video file
video = cv2.VideoCapture('TestVideo2.mp4')
# video = cv2.VideoCapture(0)


# Helper functions

#Clean noise in binary image
def clean_binary(image, threshold=600):
    labels, label_ids, values, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = values[1:, -1]; labels = labels - 1

    img2 = np.zeros((label_ids.shape),dtype = np.uint8)
    for i in range(0, labels):
        if sizes[i] >= threshold:
            img2[label_ids == i + 1] = 255
    return img2

#Increase contrast
def contrast_enhance(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(L)
    # L = cv2.equalizeHist(L)
    img_lab_merge = cv2.merge((cl, a, b))
    return cv2.cvtColor(img_lab_merge, cv2.COLOR_Lab2BGR)


def get_class_name(class_idx):
    classes = pd.read_csv('dataset/ClassNames.csv')
    class_names = classes["SignName"].values
    return class_names[class_idx]

#Merge overlapping bounding boxes
def merge_close_boxes(boxes, threshold=20):

    merged_boxes = []
    boxes = sorted(boxes, key=lambda b: b[0])  # Sort by x-coordinate

    current_box = boxes[0]
    for box in boxes[1:]:
        if (box[0] - (current_box[0] + current_box[2])) < threshold:  
            # Merge boxes
            x = min(current_box[0], box[0])
            y = min(current_box[1], box[1])
            w = max(current_box[0] + current_box[2], box[0] + box[2]) - x
            h = max(current_box[1] + current_box[3], box[1] + box[3]) - y
            current_box = (x, y, w, h)
        else:
            merged_boxes.append(current_box)
            current_box = box
    merged_boxes.append(current_box)

    return merged_boxes


#Mask for red
def mask_frame(frame):
    frame = cv2.blur(frame, (7,7))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 75, 65])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    mask = cv2.erode(mask, kernel1, iterations=1)

    mask = clean_binary(mask)

    return mask

def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_rois = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 700:  # Filter small areas
            #Estimate shape
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check for circular or triangular or octagonal shapes
            if len(approx) < 4:  #
                x, y, w, h = cv2.boundingRect(contour)
                # x = x - 15
                # y = y - 15
                # w = w + 15
                # h = h + 15
                aspect_ratio = w / float(h)
                if 0.8 < aspect_ratio < 1.2:
                    detected_rois.append((x, y, w, h))
    # detected_rois = merge_close_boxes(detected_rois)
    
    return detected_rois

def preprocess_roi(roi, target_size=(32, 32)):
    roi = cv2.resize(roi, target_size)
    roi = roi / 255.0  # Normalize
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension
    return roi



detected = []
# For each frame in video, detect potential signs
while video.isOpened():

    ret, frame = video.read()

    masked = mask_frame(frame)
    rois = find_contours(masked)
    for (x, y, w, h) in rois:
        roi = frame[y:y+h, x:x+w]
        preprocessed_roi = preprocess_roi(roi)

        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        predictions = model.predict(preprocessed_roi)
        class_id = np.argmax(predictions)
        confidence = predictions[0][class_id]

        if confidence > 0.8:  # Confidence threshold
            label = get_class_name(class_id)
            print(label)
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Road Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()