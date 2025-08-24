import numpy as np
import cv2
import tensorflow as tf
import os

# Load the trained model (adjust the filename if needed)
MODEL_PATH = r"G:\Projects\Cellula_Internship\Shop_Lifting_Classification\shopguard\detector\best_ShopGuard_model.h5"

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def create_cnn_rnn_model(input_shape, num_classes):
 # CNN base for feature extraction
 base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape[1:])
 base_model.trainable = False

 # Model architecture
 frame_input = Input(shape=input_shape)
 cnn_features = TimeDistributed(base_model)(frame_input)

 # RNN for sequence modeling
 rnn_output = GRU(64, return_sequences=False)(cnn_features)
 rnn_output = Dropout(0.5)(rnn_output)

 # Classification head
 output = Dense(num_classes, activation='softmax')(rnn_output)

 model = Model(inputs=frame_input, outputs=output)
 return model

model = create_cnn_rnn_model((20, 128, 128, 3), 2)  # Adjust input shape and num_classes as needed
model.load_weights(MODEL_PATH)

def extract_frames(video_path, max_frames=20, resize=(128, 128)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()
    return np.array(frames), frame_count

def prepare_data(video_paths, max_frames=20, resize=(128, 128)):
    X= []
    for path in zip(video_paths):
        frames, _ = extract_frames(path, max_frames, resize)
        if frames.shape[0] == max_frames:
            X.append(frames)
    return np.array(X) / 255.0

def predict_video(video_path,threshold=0.5):
    X = prepare_data(video_path)
    preds = model.predict(X)  # shape (1, num_classes)
    predicted_class = int(preds[0] > threshold)
    return predicted_class


