import cv2
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("mask_detector_model.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]
    label = "With Mask" if prediction < 0.5 else "Without Mask"
    color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

    # Show result
    cv2.putText(frame, f"{label} ({prediction:.2f})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
