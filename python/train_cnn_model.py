import os
import tensorflow as tf
from tensorflow.keras import layers, models
# dziala tylko blad indeksowania jest
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint

# Parametry
batch_size = 32
img_height = 128
img_width = 128
data_dir = os.path.join(os.getcwd(), 'Animals')

# 1. Wczytanie danych z katalogów
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
def get_class_distribution(dataset):
    all_labels = []
    for _, labels in dataset:
        all_labels.extend(labels.numpy())
    print(Counter(all_labels))

get_class_distribution(train_ds)
get_class_distribution(val_ds)

class_names = train_ds.class_names
print("Klasy:", class_names)
np.save("class_names.npy", class_names)  # zapis

# 3. Augmentacja danych tylko dla treningu
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomRotation(0.05),
])

# 2. Prefetch (przyspieszenie trenowania)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000) \
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE) \
    .cache() \
    .prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Tworzenie modelu
model = models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1. / 255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(128, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(128, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(256, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 5. Callback - wcześniejsze zatrzymanie
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

checkpoint = ModelCheckpoint("best_model.keras",
                             monitor='val_loss',
                             save_best_only=True)
with open("model_summary.json", "w") as f:
    f.write(model.to_json())


# 6. Trenowanie
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop, checkpoint])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    predicted_labels = np.argmax(preds, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Macierz pomyłek")
plt.show()


# Wykres dokładności
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.title('Dokładność')
plt.legend(loc='lower right')

# Wykres straty
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Strata')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

for i, (img, label) in enumerate(val_ds.unbatch().take(10)):
    pred = model.predict(tf.expand_dims(img, axis=0))
    print(f"Prawda: {class_names[label.numpy()]}, Predykcja: {class_names[np.argmax(pred)]}")

# 7. Zapisanie modelu
model.save("model_animals_cnn.keras")

# 8. Konwersja do TFLite
keras_model = tf.keras.models.load_model("model_animals_cnn.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

with open("model_animals_cnn.tflite", "wb") as f:
    f.write(tflite_model)

# Załaduj klasę i interpreter
class_names = np.load("class_names.npy", allow_pickle=True)
interpreter = tf.lite.Interpreter(model_path="model_animals_cnn.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Załaduj model keras
loaded_model = tf.keras.models.load_model("model_animals_cnn.keras")

# Pobierz batch z walidacji
for i, (img, label) in enumerate(val_ds.unbatch().take(10)):
    img_array = img.numpy().astype(np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)

    #  Prawdziwa etykieta
    true_class = class_names[label.numpy()]

    #  Predict z modelu po treningu (oryginalny `model`)
    keras_pred = model.predict(img_tensor, verbose=0)
    keras_class = class_names[np.argmax(keras_pred)]

    #  Predict z wczytanego modelu keras
    loaded_pred = loaded_model.predict(img_tensor, verbose=0)
    loaded_class = class_names[np.argmax(loaded_pred)]

    #  Predict z TFLite
    interpreter.set_tensor(input_details[0]['index'], img_tensor.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    tflite_class = class_names[np.argmax(output_data)]

    #  Wyniki
    print(f"{i+1}. Prawda: {true_class:8} | Trening: {keras_class:8} | Keras: {loaded_class:8} | TFLite: {tflite_class:8}")
