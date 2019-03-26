from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
#import matplotlib.pyplot as plt
print("-------------------------------------")
print(tf.__version__)
print("-------------------------------------")

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
print(" ")
print(" ")
print(" ")
print("-------------------------------------")
print(len(train_labels))
print("-------------------------------------")

print(train_labels)

print(" ")
print(" ")
print(" ")
test_images.shape

print("-------------------------------------")
print(len(test_labels))
print("-------------------------------------")

train_images = train_images / 255.0

test_images = test_images / 255.0

print(" ")
print(" ")
print(" ")
print("-----------------------------------------------")
print("-------------Configurando las capas------------")
print("-----------------------------------------------")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
print(" ")
print(" ")
print(" ")
print("-----------------------------------------------")
print("------------- Compilando el modelo ------------")
print("-----------------------------------------------")
print(" ")
print(" ")
print(" ")

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("-----------------------------------------------")
print("--------------Entrenando el modelo-------------")
print("-----------------------------------------------")
print(" ")
print(" ")
print(" ")

model.fit(train_images, train_labels, epochs=5)

print(" ")
print(" ")
print(" ")
print("-----------------------------------------------")
print("--------------Evaluando la exactitud-----------")
print("-----------------------------------------------")
print(" ")
print(" ")
print(" ")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(" ")
print(" ")
print(" ")
print('Test accuracy:', test_acc)
print(" ")
print(" ")
print(" ")
print("-----------------------------------------------")
print("--------------Haceciendo predicciones----------")
print("-----------------------------------------------")
predictions = model.predict(test_images)
print(" ")
print(predictions[0])
print(" ")
print("------------------ RESULTADO -------------------")
print(" ")
print("------------------------------------------------")
print("===   el modelo predice una etiqueta de  9   ===")
print("------------------------------------------------")
print(" ")
print("================================================")
print('PREDICCION : ', np.argmax(predictions[0]))
print("================================================")
print(" ")
print(" ")
print(" ")
print("          http://ai-trust.org (c) 2019          ")
print(" ")
print(" ")
print(" ")



