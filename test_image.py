import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("plant_disease_model.h5")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

img_path = "sample_leaf.jpg"

img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Prediction:", predicted_class)
print("Confidence:", confidence, "%")