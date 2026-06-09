import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("plant_disease_model.h5")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (128, 128))
    img_array = np.expand_dims(img, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    text = f"{predicted_class}: {confidence:.2f}%"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Plant Disease Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()