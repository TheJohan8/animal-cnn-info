import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Wczytaj model TFLite
interpreter = tf.lite.Interpreter(model_path="model_animals_cnn.tflite")
interpreter.allocate_tensors()


# Informacje o wejściu/wyjściu
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Parametry wejściowe
img_height = input_details[0]['shape'][1]
img_width = input_details[0]['shape'][2]

# Funkcja pomocnicza: wczytanie i przetworzenie zdjęcia
def load_image(path):
    img = Image.open(path).convert("RGB").resize((img_width, img_height))
    img_array = np.array(img, dtype=np.float32)   # normalizacja
    return np.expand_dims(img_array, axis=0)  # dodaj wymiar batch

# Wczytaj zdjęcie testowe
image_path = "Animals/dogs/1_0129.jpg"  # <- podmień na inne jak chcesz
input_data = load_image(image_path)
img = Image.open(image_path)
plt.imshow(img)
plt.title("Testowany obraz")
plt.axis('off')
plt.show()
print("Min/max wczytanego obrazu:", input_data.min(), input_data.max())
print("Kształt wejścia:", input_data.shape)


# Przekaż obraz do modelu
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Pobierz wynik
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output_data)

# Klasy muszą być takie same jak przy trenowaniu
class_names = np.load("class_names.npy", allow_pickle=True)
print(f"Rozpoznano: {class_names[predicted_index]}")
print("Raw output:", output_data)
print("Rozpoznano:", class_names[np.argmax(output_data)])

