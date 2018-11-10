import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sklearn
import pretty_midi
import os
import warnings
import matplotlib.pyplot as plt

"""
Implements a baseline for composer classification for our project.
Our classification accuracy among our initial database (4 composers) was 0.71875.

Much credit is due to the following Github repositorym created by Sander Shi (sandershihacker),
https://github.com/sandershihacker/midi-classification-tutorial/blob/master/midi_classifier.ipynb?fbclid=IwAR2565l9Db9J_6mJtoHqIGbCdyIQEdbMatwh7jO5S2hOrtjuTCe2jHwNWDM
upon which much of this code is based.
"""

np.random.seed(0)
composers_txt = "composers.txt"
midi_files = "NN_midi_files_extended"
percent_train = 0.6
percent_valid = 0.2
percent_test = 0.2

def load_composers(composers_txt):
	track_ids, composers = [], []
	with open(composers_txt) as f:
		while True:
			line = f.readline()
			if not line:
				break
			track_id, composer = line.strip('\n').split('\t')
			track_ids.append(track_id)
			composers.append(composer)
	return pd.DataFrame(data={"Composer": composers, "TrackID": track_ids})


def load_midi(path, composer_data):
	track_ids, midis = [], []
	for directory, composer, all_files in os.walk(path):
		files = [directory + '/' + file for file in all_files if '.mid' in file]
		for file in files:
			track_id = ''.join([str(x) for x in file if x.isdigit()])
			track_ids.append(track_id)
			midis.append(file)
	midi_data = pd.DataFrame({"TrackID": track_ids, "Path": midis})
	
	result_df = pd.merge(midi_data, composer_data, on='TrackID', how='inner')
	return result_df.drop(["TrackID"], axis=1)


def extract_features(midi_data, label_dict):
	"""
	Credit for try/except, warnings.catch_warnings, feature suggestions to sandershihacker:
	https://github.com/sandershihacker/midi-classification-tutorial/blob/master/midi_classifier.ipynb?fbclid=IwAR2565l9Db9J_6mJtoHqIGbCdyIQEdbMatwh7jO5S2hOrtjuTCe2jHwNWDM
	"""
	
	features = []
	for index, row in midi_data.iterrows():

		try:
			with warnings.catch_warnings():
				warnings.simplefilter("error")
				file = pretty_midi.PrettyMIDI(row.Path)

				tempo = file.estimate_tempo()
				num_sig_changes = len(file.time_signature_changes)
				resolution = file.resolution
				ts_changes = file.time_signature_changes
				ts_num = ts_changes[0].numerator   if len(ts_changes) > 0 else 4
				ts_den = ts_changes[0].denominator if len(ts_changes) > 0 else 4
				instruments = file.instruments
				num_instruments = len(instruments)
				composer = label_dict[row.Composer]
				
				features.append([tempo, num_sig_changes, resolution,
						ts_num, ts_den, num_instruments, composer])
		
		except:
			continue

	return np.array(features)


def features_labels(data):
	shape = data.shape[1] - 1
	features = data[:, :shape]
	labels = data[:, shape].astype(int)
	return features, labels



def data_split(data):
	data = np.random.permutation(data)
	total = len(data)
	end_train = int(percent_train * total)
	end_valid = int((percent_train + percent_valid) * total)
	train = data[: end_train]
	valid = data[end_train : end_valid]
	test  = data[end_valid:]
	return train, valid, test


def one_hot(labels, num_classes):
	return np.eye(num_classes)[labels].astype(int)


def train_model(t_features, t_labels, v_features, v_labels, num_classes):
	"""
	Credit to sandershihacker:
	https://github.com/sandershihacker/midi-classification-tutorial/blob/master/midi_classifier.ipynb?fbclid=IwAR2565l9Db9J_6mJtoHqIGbCdyIQEdbMatwh7jO5S2hOrtjuTCe2jHwNWDM
	"""
	clf_1 = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(5,), random_state=1, max_iter=600)
	clf_2 = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(5, 5), random_state=1, max_iter=600)
	clf_3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1, max_iter=600)
	clf_4 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100), random_state=1, max_iter=600)
	clf_svm = SVC()

	# Keep Track of the Best Model
	best_clf = None
	best_accuracy = 0
	
	# Test the Accuracies of the Models and Get Best
	for clf in [clf_1, clf_2, clf_3, clf_4, clf_svm]:
		t_labels_hot = one_hot(t_labels, num_classes)
		v_labels_hot = one_hot(v_labels, num_classes)
		if (type(clf) == SVC):
			clf = clf.fit(t_features, t_labels)
		else:
			clf = clf.fit(t_features, t_labels_hot)
		predictions = clf.predict(v_features)
		count = 0
		for i in range(len(v_labels)):
			if (type(clf) != SVC):
				if np.array_equal(v_labels_hot[i], predictions[i]):
					count += 1
			else:
				if v_labels[i] == predictions[i]:
					count += 1
		accuracy = count / len(v_labels_hot)
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_clf = clf

	print("Best Accuracy:", best_accuracy)
	return best_clf


def calculate_accuracy(clf, t_features, t_labels, num_classes, labels):
	"""
	Credit to sandershihacker:
	https://github.com/sandershihacker/midi-classification-tutorial/blob/master/midi_classifier.ipynb?fbclid=IwAR2565l9Db9J_6mJtoHqIGbCdyIQEdbMatwh7jO5S2hOrtjuTCe2jHwNWDM
	"""
	count = 0
	predictions = clf.predict(t_features)
	t_labels_hot = one_hot(t_labels, num_classes)
	for i in range(len(t_features)):
		if (type(clf) == SVC):
			if t_labels[i] == predictions[i]:
				count += 1
		else:
			if np.array_equal(t_labels_hot[i], predictions[i]):
				count += 1
	return count / len(t_features)

def main():
	composer_data = load_composers(composers_txt)
	labels = list(set(composer_data.Composer))
	label_dict = {label: labels.index(label) for label in labels}
	num_classes = len(labels)

	midi_data = load_midi(midi_files, composer_data)
	labeled_features = extract_features(midi_data, label_dict)
	training_set, validation_set, test_set = data_split(labeled_features)

	train_features, train_labels = features_labels(training_set)
	valid_features, valid_labels = features_labels(validation_set)
	test_features, test_labels = features_labels(test_set)

	classifier = train_model(train_features, train_labels, valid_features, valid_labels, num_classes)
	result = calculate_accuracy(classifier, test_features, test_labels, num_classes, labels)
	print(result)



if __name__ == '__main__':
	main()