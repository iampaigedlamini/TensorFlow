import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# NEURAL NETWORKS WITH SEQUENTIAL AND FUNCTIONAL API

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Sequential API (Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)
# print(model.summary())

# Alternatenative Sequential API usage
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu', name='my_layer'))
model.add(layers.Dense(10))
# print(model.summary())

#  ------ just for debugging ------
# model = keras.Model(inputs=model.inputs,
#                     outputs=[layer.output for layer in model.layers])

# features = model.predict(x_train)

# for feature in features:
#     print(feature.shape)
# ---------------------------------


# Functional API (A bit more flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

# import sys
# sys.exit()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=5)
model.evaluate(x_test, y_test, batch_size=32)
model.save('pretrain/')

