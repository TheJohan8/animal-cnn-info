import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Ścieżka do obrazu testowego
image_path = "Animals/dogs/1_0123.jpg"

# Wczytanie i przetworzenie obrazu
def load_image(path, target_size):
    img = Image.open(path).convert("RGB").resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)  # (1, h, w, 3)

# Parametry wejściowe
img_height = 128
img_width = 128
input_shape = (img_height, img_width)

input_image = load_image(image_path, input_shape)

# ======================= Model Keras =======================
model_keras = tf.keras.models.load_model("model_animals_cnn.keras")
output_keras = model_keras.predict(input_image)
predicted_keras = np.argmax(output_keras)

# ======================= Model TFLite =======================
interpreter = tf.lite.Interpreter(model_path="model_animals_cnn.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_image.astype(np.float32))
interpreter.invoke()
output_tflite = interpreter.get_tensor(output_details[0]['index'])
predicted_tflite = np.argmax(output_tflite)

# ======================= Wyświetlenie wyników =======================
class_names = np.load("class_names.npy", allow_pickle=True)

plt.imshow(Image.open(image_path))
plt.title("Testowany obraz")
plt.axis("off")
plt.show()

print("==== WYNIKI PORÓWNANIA ====")
print(f"Keras prediction index:   {predicted_keras} ({class_names[predicted_keras]})")
print(f"TFLite prediction index:  {predicted_tflite} ({class_names[predicted_tflite]})")

print("\nRaw output (Keras):", output_keras)
print("Raw output (TFLite):", output_tflite)
