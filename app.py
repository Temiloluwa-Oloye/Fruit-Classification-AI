"""
Fruit Classifier Application
Main entry point for the fruit classification application.
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_classifier():
    """Load the trained model and class indices."""
    model_path = os.path.join('models', 'fruit_classifier_model.h5')
    config_path = os.path.join('config', 'class_indices.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Class indices not found at {config_path}. Please generate them first.")
    
    model = load_model(model_path)
    
    with open(config_path, 'r') as f:
        class_indices = json.load(f)
    
    return model, class_indices

def preprocess_image(image_path, target_size=(64, 64)):
    """Preprocess an image for prediction."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fruit(model, class_indices, image_path):
    """
    Predict the fruit class from an image.
    
    Args:
        model: Loaded Keras model
        class_indices: Dictionary mapping class indices to fruit names
        image_path: Path to the image file
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction, axis=-1)[0]
    confidence = float(prediction[0][predicted_index])
    predicted_class = class_indices.get(str(predicted_index), "Unknown")
    
    return predicted_class, confidence

if __name__ == "__main__":
    print("Loading fruit classifier model...")
    try:
        model, class_indices = load_classifier()
        print("Model loaded successfully!")
        print(f"Number of classes: {len(class_indices)}")
        print("\nAvailable fruit classes:")
        for idx, fruit_name in sorted(class_indices.items(), key=lambda x: int(x[0])):
            print(f"  {idx}: {fruit_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
