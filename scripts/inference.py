import cv2
import os

import warnings
warnings.filterwarnings('ignore')

import os
from timeit import time
import sys
import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
#from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
#from deep_sort.detection import Detection as ddet

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 1000

MAX_SEQ_LENGTH = 25
NUM_FEATURES = 2048


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


feature_extractor = build_feature_extractor()

label_processor = tf.keras.layers.experimental.preprocessing.StringLookup(num_oov_indices=0, vocabulary=np.unique(['', 'Afraid', 'After', 'Afternoon', 'Again', 'Alawys', 'All', 'Always', 'Angry', 'Animal', 'Answer', 'Apple', 'Argue', 'Art', 'Ask', 'Aunt', 'Autumn', 'Awake', 'Baby', 'Back', 'Bad', 'Ball', 'Bear', 'Because', 'Best', 'Better', 'Bicycle', 'Big', 'Bird', 'Birth', 'Biscuit', 'Black', 'Blood', 'Blue', 'Boat', 'Body', 'Book', 'Both', 'Box', 'Boy', 'Bread', 'Breathe', 'Bring', 'Brother', 'Brown', 'Bug', 'Bus', 'But', 'Buy', 'Can', 'Cannot', 'Car', 'Carry', 'Cat', 'Chair', 'Child', 'Choose', 'Climb', 'Closet', 'Clothes', 'Cloud', 'Coat', 'Cochlear Implant', 'Cold', 'Colour', 'Computer', 'Cookie', 'Correct', 'Count', 'Crazy', 'Cup', 'Dance', 'Dark']))

yolo_path = r"C:\TelAviv\D5"

sequence_model = tf.keras.models.load_model("D:\hack\sl_model.h5")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(1, 3),
	dtype="uint8")

for fname in os.listdir(yolo_path):
    if('.names' in fname):
        names_data = os.path.join(yolo_path,fname)
    if('.cfg' in fname):
        cfg_data = os.path.join(yolo_path,fname)
    if('.weights' in fname):
        weights_data = os.path.join(yolo_path,fname)

net = cv2.dnn.readNetFromDarknet(cfg_data, weights_data)




cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    (H, W) = frame.shape[:2]
    # determine only the output layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (224, 224),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
    	# loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.1:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        try:
            if len(idxs) > 0:
            # loop over the indexes we are keeping
                cropped = []
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cropped_img = frame[x:x+w,y:y+h]
                    cropped_img = crop_center_square(cropped_img)
                    cropped_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
                    cropped_img = cropped_img[:, :, [2, 1, 0]]

                    cropped.append(cropped_img)
        except:
            pass



            class_vocab = label_processor.get_vocabulary()
             
            frame_features, frame_mask = prepare_single_video(np.array(cropped))
            probabilities = sequence_model.predict([frame_features, frame_mask])[0]
            pred_class_id = np.argsort(probabilities)[::-1][0]
            print(class_vocab[pred_class_id])
            cv2.putText(frame, class_vocab[pred_class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # for i in np.argsort(probabilities)[::-1]:
            #     print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")


        cv2.imshow("Image", frame)









    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
