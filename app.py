from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

# Load the model
model = None
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model():
    global model
    try:
        # Try to load saved model
        if os.path.exists('cifar10_model.h5'):
            model = keras.models.load_model('cifar10_model.h5')
            print("✓ Model loaded from cifar10_model.h5")
        else:
            print("⚠ Model file not found. Creating a simple model for testing...")
            print("  To use your trained model, run: model.save('cifar10_model.h5')")
            # Create a dummy model for testing
            model = keras.Sequential([
                keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Conv2D(64, (3,3), activation='relu'),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print("✓ Dummy model created")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.route('/')
def index():
    """Serve the HTML file"""
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Resize to 32x32 (CIFAR-10 size)
        img = img.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Return results
        return jsonify({
            'predicted_class': int(predicted_class),
            'predicted_class_name': class_names[predicted_class],
            'confidence': confidence,
            'predictions': predictions[0].tolist()
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    """Get model information"""
    return jsonify({
        'model': 'CIFAR-10 Classifier',
        'classes': class_names,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Loading CIFAR-10 Classification Model")
    print("=" * 50)
    load_model()
    print("\n✓ Starting Flask server...")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5000)
