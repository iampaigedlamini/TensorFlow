import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

'''
# =============================================================== #
#                         Pre-trained Model
# =============================================================== #
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = keras.models.load_model('pretrain/')
model.trainable = False

for layer in model.layers:
    assert layer.trainable == True
    layer.trainable = False
    
# get layers by index
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_outputs = layers.Dense(10)(base_outputs)

new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)
# print(new_model.summary())

new_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

new_model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=2)
new_model.evaluate(x_test, y_test, batch_size=64)
'''

'''
# =============================================================== #
#                      Pre-trained Keras Model
# =============================================================== #

x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

model = keras.applications.InceptionV3(include_top=True)
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_outputs = layers.Dense(5)(base_outputs)

new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)

new_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

new_model.fit(x, y, batch_size=64, epochs=15, verbose=2)
#new_model.evaluate(x, y, batch_size=64)
'''


# =============================================================== #
#                       Pre-trained Hub Model
# =============================================================== #
x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

url = 'https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1'

base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))
base_model.trainable =False

model = keras.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu',),
    layers.Dense(5)
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model.fit(x, y, batch_size=64, epochs=15)




