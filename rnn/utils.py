# Inspired by https://github.com/brannondorsey/midi-rnn/blob/master/train.py


import numpy as np
import os
from collections import defaultdict
import pretty_midi

np.random.seed(1)

# cutoff = 1000
piano_roll_cutoff = 1536 # 1024
window_size = 1024 # 512
frequency = 20 # 10


def get_params():
	return piano_roll_cutoff, window_size, frequency


def parse_midi(path):
	midi = None
	try:
		midi = pretty_midi.PrettyMIDI(path)
		midi.remove_invalid_notes()
	except Exception as e:
		raise Exception(("%s\nerror readying midi file %s" % (e, path)))
	return midi

def zero_pad(arr, x, missing):
	right = np.zeros((x, missing))
	return np.hstack((arr,right))
	

def split_arrays(arr):
	x,y = arr.shape
	if y < (piano_roll_cutoff):
		return [zero_pad(arr, x, (piano_roll_cutoff) - y)]

	arrs = []
	while True:
		arrs.append(arr[:,:piano_roll_cutoff])
		arr = arr[:,window_size:]
		if arr.shape[1] < piano_roll_cutoff:
			arrs.append(arr)
			return arrs




def separate_data(dirname_arg):
	this_dir_name = '../data/sets'
	directory = os.fsencode(this_dir_name)
	
	i = 0
	result_set = False
	result = None
	labels = []

	for dirname in [this_dir_name]:
		new_dir = this_dir_name
		
		for file in os.listdir(new_dir):
			filename = os.fsdecode(file)
			if filename[0] == '.':
				continue
			print(filename) # for tracking purposes only
			
			composer = ''.join([i for i in filename[:-4] if not i.isdigit()])
			midi = parse_midi(new_dir + '/' + filename)

			if midi is not None:
				
				arrays = split_arrays(midi.get_piano_roll(fs=frequency))
				for arr in arrays:
					if arr.shape != (128, piano_roll_cutoff):
						# optional error handling
						pass
					elif not result_set:
						result = arr.reshape((1,arr.shape[0],arr.shape[1]))
						labels.append(composer)
						result_set = True
					else:
						result = np.vstack((result, arr.reshape((1,arr.shape[0],arr.shape[1]))))
						labels.append(composer)


	# Randomize data
	indices = np.arange(result.shape[0])
	np.random.shuffle(indices)
	result = result[indices]
	labels = list(np.asarray(labels)[indices])
	return result, labels


def load_dataset():

	print("Collecting Data.....")
	data, labels = separate_data('data/sets')
	print("Data collected.")	


	cut = int(.85 * len(labels))
	train = data[:cut]
	train_labels = labels[:cut]
	test = data[cut:]
	test_labels = labels[cut:]

	classes = list(set(train_labels + test_labels))
	classes_map = defaultdict(int)
	for i,x in enumerate(classes):
		classes_map[x] = i


	def one_hot(orig):
		result = np.zeros((orig.size, len(classes)+1))
		for i in range(orig.size):
			result[i][orig[i]] = 1
		return result

	def get_X(orig):
		result = np.zeros((len(orig), cutoff))
		for i in range(len(orig)):
			result[i] = orig[i][0]
		return result.reshape(len(orig), cutoff, 1)

	train_labels = one_hot(np.array([classes_map[x] for x in train_labels]).reshape(len(train),1))
	test_labels = one_hot(np.array([classes_map[x] for x in test_labels]).reshape(len(test),1))

	return train, train_labels, test, test_labels
