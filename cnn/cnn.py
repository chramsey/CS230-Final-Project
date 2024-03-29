### CNN implementation

import numpy as np
import util
import keras
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, GlobalAveragePooling1D
import matplotlib.pyplot as plt

# Hyperparameters
sample_len = 1024
train_sample_p = 0.1
test_sample_p = 0.5
test_sample_cap = 30
batch_size = 128
epochs = 50
reg_l2 = 0.05
conv_dropout = 0.25
dense_dropout = 0.5
model_name = './model.h5'

X_train, Y_train, _, _, count = util.load_even_dataset('../NN_midi_files_extended/train/', sample_len, train_sample_p)
X_dev, Y_dev, composer_to_index, index_to_composer, count_dev = util.load_even_dataset('../NN_midi_files_extended/dev/', sample_len, train_sample_p)

print X_train.shape
print Y_train.shape
print X_dev.shape
print Y_dev.shape
print index_to_composer
print count
print count_dev

num_classes = len(index_to_composer)

model = Sequential()
model.add(Conv1D(32, 3, strides=3,
                 input_shape=X_train.shape[1:], kernel_regularizer=regularizers.l2(reg_l2)))
model.add(Activation('tanh'))
model.add(Conv1D(32, 1, kernel_regularizer=regularizers.l2(reg_l2)))
model.add(Activation('tanh'))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(64, 3, kernel_regularizer=regularizers.l2(reg_l2)))
model.add(Activation('tanh'))
model.add(Conv1D(64, 1, kernel_regularizer=regularizers.l2(reg_l2)))
model.add(Activation('tanh'))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(128, 3, kernel_regularizer=regularizers.l2(reg_l2)))
model.add(Activation('tanh'))
model.add(Conv1D(128, 1, kernel_regularizer=regularizers.l2(reg_l2)))
model.add(Activation('tanh'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(reg_l2)))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              # validation_split=0.01,
              validation_data=(X_dev, Y_dev),
              shuffle=True)

model.save(model_name)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model = load_model(model_name)
accuracy, short_files = util.test_dataset('../NN_midi_files_extended/test', model, sample_len, test_sample_p, test_sample_cap)
print accuracy
print short_files
print 'Done...'

