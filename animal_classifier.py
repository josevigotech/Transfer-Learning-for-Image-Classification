# Proyectos para Puerto Rico
# Descomprimir archivos y preparar dataset
!unzip serpiente_reticulada/serpiente_reticulada.zip -d serpiente_reticulada
!unzip caimanes/caimanes.zip -d caimanes
!rm -rf caiman/caiman.zip
!rm -rf serpiente_reticulada/serpiente_reticulada.zip
# Crear carpeta dataset y copiar imágenes
!mkdir dataset
!cp -r caimanes dataset/caimanes
!cp -r serpiente_reticulada dataset/serpiente_reticulada

# Aumento de datos
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=5,
    zoom_range=[0.7, 1.3],
    validation_split=0.2
)

data_gen_entrenamiento = datagen.flow_from_directory(
    "/content/dataset",
    target_size=(224,224),
    batch_size=32,
    shuffle=True,
    subset="training"
)

data_gen_prueba = datagen.flow_from_directory(
    "/content/dataset",
    target_size=(224,224),
    batch_size=32,
    shuffle=True,
    subset="validation"
)

import matplotlib.pyplot as plt

for imagenes, etiquetas in data_gen_entrenamiento:
  for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(imagenes[i])
  break
plt.show()

import tensorflow as tf
import tensorflow_hub as hub

mobilenetv2 = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

mobilenetv2.trainable = False

modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation="softmax")
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
EPOCAS = 20
entrenamiento = modelo.fit(
    data_gen_entrenamiento,
    epochs=EPOCAS,
    batch_size=32,
    validation_data=data_gen_prueba
)

from PIL import Image
import numpy as np

def categorizar(ruta):
    imagen = Image.open(ruta).convert("RGB")
    imagen = imagen.resize((224, 224))
    imagen_array = np.array(imagen) / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)

    prediccion = modelo.predict(imagen_array)
    clase_predicha = np.argmax(prediccion, axis=1)[0]

    if clase_predicha == 0:
        return "caimanes", 0
    elif clase_predicha == 1:
        return "serpiente_reticulada", 1
    else:
        return "Clase desconocida", -1

ruta = "caimanes.jpg"
etiqueta, valor = categorizar(ruta)
print(f"Etiqueta: {etiqueta}, Valor: {valor}")

ruta = "serpiente.jpg"
etiqueta, valor = categorizar(ruta)
print(f"Etiqueta: {etiqueta}, Valor: {valor}")
