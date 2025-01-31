import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the model (this should be the same model saved in train.py)
model = tf.keras.models.load_model('image_classifier_model.h5')

# Function to predict image class
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust to your model's expected input size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    return predicted_class_index

# Example usage (for testing outside the web app):
if __name__ == '__main__':
    image_path = 'path_to_your_image.jpg'
    predicted_class_index = predict_image(image_path)
    print(f"Predicted class index: {predicted_class_index}")
