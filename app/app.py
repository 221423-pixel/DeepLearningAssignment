import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__, static_folder='../static', template_folder='../templates')

# Configuration
MODEL_PATH = os.path.abspath("app/models/action_recognition_model.h5")
CLASS_INDICES_PATH = os.path.abspath("app/models/class_indices.json")

# Load Model and Indices
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

print("Loading class indices...")
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
        # Convert keys to int if they are strings (JSON keys are always strings)
        # But our values are class names (strings), keys are indices (integers)
        # Wait, in train_model.py: inverted_indices = {v: k for k, v in class_indices.items()}
        # So key is index (str in json), value is class name.
        # Let's ensure keys are integers for lookup
        idx_to_class = {int(k): v for k, v in class_indices.items()}
    print("Class indices loaded successfully.")
except Exception as e:
    print(f"Error loading class indices: {e}")
    idx_to_class = {}

def prepare_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Read image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image)
        
        # Predict
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        predicted_class = idx_to_class.get(predicted_class_idx, "Unknown")
        
        return jsonify({
            'action': predicted_class,
            'confidence': confidence,
            'all_predictions': {idx_to_class[i]: float(predictions[0][i]) for i in range(len(idx_to_class))}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
