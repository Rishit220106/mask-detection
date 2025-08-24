import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Load the saved model
model = tf.keras.models.load_model("mask_detector_model.keras")  # or 'mask_detector_model.h5' if you saved as HDF5

# Image path (you can pass it from command line or hardcode it)
img_path = sys.argv[1] if len(sys.argv) > 1 else "data/without.jpg"

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
prediction = model.predict(img_array)
print("Raw model output:", prediction[0][0])

# Interpret result (fixed logic to match detect.py)
if prediction[0][0] < 0.5:
    print("Prediction: With Mask 😷")
    confidence = 1 - prediction[0][0]
else:
    print("Prediction: Without Mask 😷")
    confidence = prediction[0][0]

print(f"Confidence: {confidence:.4f}")