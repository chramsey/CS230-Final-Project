# Inspired by https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification/blob/master/lstm_genre_classifier_keras.py

import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TerminateOnNaN, EarlyStopping, History
from keras import regularizers
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import utils



np.random.seed(1)

plot_folder = str(datetime.date.today()) + '_' + str(datetime.datetime.now().time()) + '/'
os.mkdir(plot_folder[:-1])
output_file = plot_folder + 'output'



# HYPERPARAMS
batch_size = 24 # 35
nb_epochs = 100
learning_rate = 0.01
dropout_layer1 = 0.4
dropout_layer2 = 0.4
dropout_layer3 = 0.4
patience = 3 # How many chances it has to increase accuracy
validation_size = 0.2 # What percentage of the training set should be reserved for val testing
l2_reg = 0.01


train_X, train_Y, test_X, test_Y = utils.load_dataset()

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
# opt = Adam(lr=0.01, clipnorm=1.)

with open(output_file, 'w+') as f:
	piano_cutoff, window_size, fs = utils.get_params()
	f.write('Piano roll cutoff: ' + str(piano_cutoff) + '\n')
	f.write('Window size: ' + str(window_size) + '\n')
	f.write('Frequency of samples: ' + str(fs) + '\n')
	f.write('\n')

# for _ in range(20):
while True:
	print()
	print()

	batch_size_input = input('Batch size? Default = 24. ')
	if batch_size_input:
		try: 
			batch_size = int(batch_size_input)
		except ValueError:
			pass    
	print('\tBatch size set to ' + str(batch_size))

	learning_rate_input = input('Learning rate? Default = 0.01. ')
	if learning_rate_input:
		try:
			learning_rate = float(learning_rate_input)
		except ValueError:
			pass
	print('\tLearning rate set to ' + str(learning_rate))

	d1 = input('Dropout of first layer? Default = 0.4. ')
	d2 = input('Dropout of second layer? Default = 0.4. ')
	d3 = input('Dropout of third layer? Default = 0.4. ')
	try:
		d1 = float(d1)
		dropout_layer1 = float(d1)
	except ValueError:
		pass
	try:
		d2 = float(d2)
		dropout_layer2 = float(d2)
	except ValueError:
		pass
	try:
		d3 = float(d3)
		dropout_layer3 = float(d3)
	except ValueError:
		pass
	print('\tDropout for layer 1: ' + str(dropout_layer1) + '\n' \
		  '\tDropout for layer 2: ' + str(dropout_layer2) + '\n' \
		  '\tDropout for layer 3: ' + str(dropout_layer3))

	patience_input = input('Patience? Default = 3. ')
	try:
		patience_input = int(patience_input)
		patience = patience_input
	except ValueError:
		pass
	print('\tPatience set to ' + str(patience))

	validation_size_input = input('Percent of training data reserved for validation? Default = 0.1. ')
	try:
		validation_size_input = float(validation_size_input)
		validation_size = validation_size_input
	except:
		pass
	print('\tValidation size set to ' + str(validation_size))

	l2_reg_input = input('L2 regularization? Default = 0.01. ')
	try:
		l2_reg_input = float(l2_reg_input)
		l2_reg = l2_reg_input
	except ValueError:
		pass
	print('\tL2 Regularization set to ' + str(l2_reg))

	# # If you want to run with random numbers in some range, comment out all code from the
	# #    while True loop to here and uncomment the following lines. Also add back the for loop
	# batch_size = random.choice([1,16,32,64])
	# learning_rate = random.choice([0.1, 0.05, 0.5, 0.25])
	# dropout_layer1 = random.choice([0.3,0.4,0.5])
	# dropout_layer2 = random.choice([0.3,0.4,0.5])
	# dropout_layer3 = random.choice([0.3,0.4,0.5])
	# l2_reg = random.choice([0.1, 0.05, 0.01, 0.5])
	# layers = random.choice([2,3,4])



	with open(output_file, 'a') as f:
		f.write('=======================================\n')
		f.write('=======================================\n')
		f.write('=======================================\n')
		f.write('Batch Size: ' + str(batch_size) + '\n')
		f.write('Learning Rate: ' + str(learning_rate) + '\n')
		f.write('Dropout: ' + str(dropout_layer1) + ', ' + str(dropout_layer2) + ', ' + str(dropout_layer3) + '\n')
		f.write('L2 regularization: ' + str(l2_reg) + '\n')
		f.write('Layers: ' + str(layers))
		f.write('\n')


	opt = Adam(lr=learning_rate)

	nan = TerminateOnNaN()
	es = EarlyStopping(monitor='val_acc', patience=patience)



	print("Training X shape: " + str(train_X.shape))
	print("Training Y shape: " + str(train_Y.shape))
	# Dev is part of training set
	print("Test X shape: " + str(test_X.shape))
	print("Test Y shape: " + str(test_X.shape))

	input_shape = (train_X.shape[1], train_X.shape[2])
	print('Build LSTM RNN model ...')
	model = Sequential()
	model.add(LSTM(units=128, dropout=dropout_layer1, return_sequences=True, input_shape=input_shape))
	if layers > 2:
		model.add(LSTM(units=128, dropout=dropout_layer2, return_sequences=True))
	if layers > 3:
		model.add(LSTM(units=128, dropout=dropout_layer2, return_sequences=True))
	model.add(LSTM(units=32, dropout=dropout_layer3, return_sequences=False))

	model.add(Dense(units=train_Y.shape[1], activation='softmax', kernel_regularizer=regularizers.l2(l2_reg)))

	print("Compiling ...")
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()

	print("Training ...")
	history = model.fit(train_X, 
						  train_Y, 
						  batch_size=batch_size, 
						  epochs=nb_epochs, 
						  validation_split=validation_size, 
						  callbacks=[nan, es])


	with open(output_file, 'a') as f:
		f.write('\n')
		f.write('Training Summary:\n')
		for key in history.history.keys():
			f.write(key + ':\n')
			val = history.history[key]
			if isinstance(val, (list,)):
				for epoch, elem in enumerate(val):
					f.write('\t' + str(epoch) + ') ' + str(elem) + '\n')
			else:
				f.write(val)
			f.write('\n')
		f.write('\n')

	print("\nTesting ...")
	score, accuracy = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=1)
	print("Test loss:  ", score)
	print("Test accuracy:  ", accuracy)

	keys = history.history.keys()
	if 'acc' not in keys or 'val_acc' not in keys or 'loss' not in keys or 'val_loss' not in keys:
		continue
	else:

		# Accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plot_name = plot_folder + str(_) + 'Accuracy.jpg'
		plt.savefig(plot_name)
		plt.clf()

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plot_name = plot_folder + str(_) + 'Loss.jpg'
		plt.savefig(plot_name)
		plt.clf()


	with open(output_file, 'a') as f:
		f.write('\n')
		f.write('Test Summary:\n')
		f.write('Score: ' + str(score) + '\n')
		f.write('Accuracy: ' + str(accuracy) + '\n')
		f.write('\n')
