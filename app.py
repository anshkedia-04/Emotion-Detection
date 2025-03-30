from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels (Ensure these match your training labels)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def preprocess_image(image_data):
    """Preprocess image before feeding into the model."""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (48, 48))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=[0, -1])  # Add batch and channel dimensions
    return img

@app.route("/")
def index():
    """Render the frontend."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Receive an image from frontend, process it, and return emotion prediction."""
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])  # Decode Base64

    img = preprocess_image(image_data)
    
    prediction = model.predict(img)
    emotion = emotion_labels[np.argmax(prediction)]  # Get the predicted label

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)
