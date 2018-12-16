import numpy as np
import os
from collections import defaultdict
import pretty_midi

def parse_midi(path):
	midi = None
	try:
		midi = pretty_midi.PrettyMIDI(path)
		midi.remove_invalid_notes()
	except Exception as e:
		raise Exception(("%s\nerror readying midi file %s" % (e, path)))
	return midi

def read_midi_files(root):
	print 'Loading dataset into [(midi, composer), (midi, composer), ...]'
	data = []
	composer_index = 0
	composer_to_index = {}
	index_to_composer = []
	for root, directories, filenames in os.walk(root):
		if len(filenames) > 1: # ignore .DS_Store
			composer = root.split('/')[-1]
			composer_to_index[composer] = composer_index
			index_to_composer.append(composer)
			for filename in filenames:
				if filename.endswith('.mid'):
					print 'new midi ...... %s' % os.path.join(root,filename)
					midi = parse_midi(os.path.join(root,filename))
					data.append((midi, composer_index))
			composer_index += 1
	return data, composer_to_index, index_to_composer

def sample_midi(midi, l=256, p=0.01):
	midi_piano_roll = midi.get_piano_roll(fs=10)
	if midi_piano_roll.shape[1] < l:
		return []
	samples = []
	sample_pool = range(midi_piano_roll.shape[1] - l + 1)
	sample_size = int(np.ceil(len(sample_pool) * p))
	sample_starts = np.random.choice(sample_pool, size=sample_size, replace=False)
	for start in sample_starts:
		sample = midi_piano_roll[:, start: start + l]
		samples.append(np.reshape(sample, (sample.shape[1], sample.shape[0])))
	return np.asarray(samples)

def sample_midi_start(midi, l=256, p=0.01):
	midi_piano_roll = midi.get_piano_roll(fs=10)
	if midi_piano_roll.shape[1] < l:
		return []
	samples = []
	sample_pool = range(midi_piano_roll.shape[1] - l + 1)
	sample_size = int(np.ceil(len(sample_pool) * p))
	sample_starts = np.random.choice(sample_pool, size=sample_size, replace=False)
	for start in sample_starts:
		sample = start
		samples.append(start)
	return np.asarray(samples)

def sample_midi_even(midi, l=256, p=0.5, c=30):
	midi_piano_roll = midi.get_piano_roll(fs=10)
	if midi_piano_roll.shape[1] < l:
		return []
	samples = []
	sample_pool = range(midi_piano_roll.shape[1] - l + 1)
	sample_size = min(c, int(np.ceil(len(sample_pool) * p)))
	sample_starts = np.random.choice(sample_pool, size=sample_size, replace=False)
	for start in sample_starts:
		sample = midi_piano_roll[:, start: start + l]
		samples.append(np.reshape(sample, (sample.shape[1], sample.shape[0])))
	return np.asarray(samples)

def load_dataset(root, l, p):
	data, composer_to_index, index_to_composer = read_midi_files(root)
	num_composer = len(index_to_composer)
	count = np.zeros(num_composer, dtype=int)
	X = None
	Y = None
	n = 0
	for midi, composer_index in data:
		print 'new midi sampling......%d' % n
		n += 1
		samples = sample_midi(midi, l, p)
		if len(samples) != 0:
			one_hot = np.zeros((len(samples), num_composer))
			one_hot[:, composer_index] = 1
			count[composer_index] += samples.shape[0]
			if X is None:
				X = samples
				Y = one_hot
			else:
				X = np.vstack((X, samples))
				Y = np.vstack((Y, one_hot))
	return X, Y, composer_to_index, index_to_composer, count

def list_tup_to_dict(lst):
	d = {}
	for v, k in lst:
		if k in d:
			d[k].append(v)
		else:
			d[k] = [v]
	return d

def load_even_dataset(root, l, p):
	data, composer_to_index, index_to_composer = read_midi_files(root)
	num_composer = len(index_to_composer)
	count = np.zeros(num_composer, dtype=int)
	n = 0
	for midi, composer_index in data:
		print 'new midi sampling......%d' % n
		n += 1
		samples = sample_midi(midi, l, p)
		if len(samples) != 0:
			count[composer_index] += samples.shape[0]
	min_count = min(count)
	X = [None] * num_composer
	n = 0
	data = list_tup_to_dict(data)
	for composer_index in data:
		midis = data[composer_index]
		for midi in midis:
			print 'new midi sampling......%d' % n
			n += 1
			samples = sample_midi(midi, l, p)
			if len(samples) != 0:
				if X[composer_index] is None:
					X[composer_index] = samples
				else:
					X[composer_index] = np.vstack((X[composer_index], samples))
		X[composer_index] = X[composer_index][np.random.choice(range(count[composer_index]), size=min_count, replace=False)]
	X_even = None
	Y = None
	for composer_index in range(num_composer):
		one_hot = np.zeros((min_count, num_composer))
		one_hot[:, composer_index] = 1
		if X_even is None:
			X_even = X[composer_index]			
			Y = one_hot
		else:
			X_even = np.vstack((X_even, X[composer_index]))
	 		Y = np.vstack((Y, one_hot))
	idx = np.random.permutation(num_composer * min_count)
	return X_even[idx], Y[idx], composer_to_index, index_to_composer, count

def test_dataset(root, model, l, p, c):
	data, _, index_to_composer = read_midi_files(root)
	num_composer = len(index_to_composer)
	short_files = 0
	correct = 0
	for midi, composer_index in data:
		samples = sample_midi_even(midi, l, p, c)
		if len(samples) != 0:
			print len(samples)
			vote = np.sum(model.predict(samples),axis=0)
			if np.argmax(vote) == composer_index:
				correct += 1
			
			# votes = np.argmax(model.predict(samples), axis=1)
			# count = [0] * num_composer
			# for vote in votes:
			# 	count[vote] += 1
			# if np.argmax(count) == composer_index:
			# 	correct += 1
		else:
			short_files += 1
	return float(correct)/len(data), short_files
