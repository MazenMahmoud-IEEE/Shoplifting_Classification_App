# detector/views.py
import os
import numpy as np
import cv2
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, Dropout, Input
from tensorflow.keras.models import Model

# ---------------------
# Load model
# ---------------------
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "detector",
    "model",
    "best_ShopGuard_model.h5"
)

def create_cnn_rnn_model(input_shape, num_classes):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=input_shape[1:]
    )
    base_model.trainable = False

    frame_input = Input(shape=input_shape)
    cnn_features = TimeDistributed(base_model)(frame_input)

    rnn_output = GRU(64, return_sequences=False)(cnn_features)
    rnn_output = Dropout(0.5)(rnn_output)

    output = Dense(num_classes, activation='softmax')(rnn_output)

    model = Model(inputs=frame_input, outputs=output)
    return model

model = create_cnn_rnn_model((20, 128, 128, 3), 2)
model.load_weights(MODEL_PATH)

# ---------------------
# Video preprocessing
# ---------------------
def extract_frames(video_path, max_frames=20, resize=(128, 128)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, frame_count // max_frames)  # spread frames across video

    for i in range(0, frame_count, step):
        if len(frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()
    return np.array(frames)

def prepare_data(video_paths, max_frames=20, resize=(128, 128)):
    X = []
    for path in video_paths:
        frames = extract_frames(path, max_frames, resize)
        if frames.shape[0] == max_frames:
            X.append(frames)
    return np.array(X) / 255.0

# ---------------------
# Django view
# ---------------------
def index(request):
    context = {}
    if request.method == "POST" and request.FILES.get("video"):
        video = request.FILES["video"]

        # Save video in MEDIA_ROOT/uploads
        fs = FileSystemStorage(location='media/uploads')
        filename = fs.save(video.name, video)
        video_path = fs.path(filename)

        # Preprocess & predict
        x = prepare_data([video_path])
        prediction = model.predict(x)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Convert prediction into label
        if predicted_class == 0:
            result = "Shoplifting Detected 🚨"
        else:
            result = "Normal Behavior ✅"

        context["result"] = result
        context["file_url"] = fs.url(filename)
    return render(request, "detector/index.html", context)