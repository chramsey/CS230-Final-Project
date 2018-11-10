import tensorflow as tf
from tensorflow import keras
import numpy as np
import midi_parser

data = midi_parser.parse_data()
X = []
Y = []
for (x, y) in data:
	X.append(x)
	Y.append(y)
	
model = keras.Sequential([
    keras.layers.dense(input_shape=(10000,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, Y, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
