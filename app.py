'''from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
from sklearn.metrics import accuracy_score, precision_score

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('./model/deepfake_video_model.h5')

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Feature extractor using InceptionV3
def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Function to load and process video frames
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
            frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Crop center square of frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Prepare video for model
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = valid frame, 0 = masked

    return frame_features, frame_mask

@app.route('/')
def home():
    return render_template('indexf.html')

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)
    
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames)
    
    prediction = model.predict([frame_features, frame_mask])[0]
    predicted_label = 1 if prediction >= 0.51 else 0  # 1 = FAKE, 0 = REAL
    
    # **True Label Handling (Modify based on dataset)**
    true_label = 1 if "fake" in video.filename.lower() else 0  # Modify for real data

    # Compute Accuracy & Precision (Assuming multiple videos)
    global true_labels, predicted_labels
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)

    os.remove(video_path)  # Clean up video after processing
    
    return jsonify({
        'result': 'FAKE' if predicted_label else 'REAL',
        'confidence': float(prediction),
        'accuracy': f"{round(accuracy * 100, 2)}%",  # Convert to percentage
        'precision': f"{round(precision * 100, 2)}%"  # Convert to percentage
    })

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Initialize label lists
    true_labels = []
    predicted_labels = []

    app.run(debug=True)'''
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
from collections import deque
from sklearn.metrics import accuracy_score, precision_score

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('./model/deepfake_video_model.h5')

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Store recent predictions for metrics (limited to last 20 samples)
RECENT_PREDICTIONS = deque(maxlen=20)

# Feature extractor using InceptionV3
def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Function to load and process video frames
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
            frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Crop center square of frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Prepare video for model
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = valid frame, 0 = masked

    return frame_features, frame_mask

@app.route('/')
def home():
    return render_template('index2.html')

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)
    
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames)
    
    prediction = model.predict([frame_features, frame_mask])[0]
    confidence = float(prediction)
    predicted_label = 1 if confidence >= 0.51 else 0  # 1 = FAKE, 0 = REAL
    
    # Get true label from filename (for demo purposes - in production use proper labels)
    true_label = 1 if "fake" in video.filename.lower() else 0
    
    # Store the prediction for metrics
    RECENT_PREDICTIONS.append((true_label, predicted_label))
    
    # Calculate metrics if we have enough samples
    metrics = {}
    if len(RECENT_PREDICTIONS) >= 5:  # Only show metrics after 5 samples
        true_labels, pred_labels = zip(*RECENT_PREDICTIONS)
        metrics['accuracy'] = f"{accuracy_score(true_labels, pred_labels)*100:.1f}%"
        metrics['precision'] = f"{precision_score(true_labels, pred_labels, zero_division=0)*100:.1f}%"
    else:
        metrics['accuracy'] = "Not enough data (min 5 samples)"
        metrics['precision'] = "Not enough data (min 5 samples)"
    
    os.remove(video_path)  # Clean up video after processing
    
    return jsonify({
        'result': 'FAKE' if predicted_label else 'REAL',
        'confidence': confidence,
        **metrics
    })

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
