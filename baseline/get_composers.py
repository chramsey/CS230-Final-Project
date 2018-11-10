import os

"""
Creates a labeled file matching a unique track_id to a composer,
based off of the names of the files in the dataset.

Means to be called immediately following rename_files.py and before
baseline_svm.py.
"""


dataset = "NN_midi_files_extended"

def main():
	info = []
	used = []
	track_id = 0
	for dir_name, subdir_list, file_list in os.walk(dataset):
		print(dir_name)
		print(file_list)
		file_path_list = ['/'.join([dir_name, file]) for file in file_list if '.mid' in file]
		for file_path in file_list:
			print(file_path)
			if '.mid' not in file_path: # or ('bwv' not in file_path and 'byrd' not in file_path and 'chet' not in file_path):
				continue
			track_id = ''.join([str(x) for x in file_path if x.isdigit()])
			composer = file_path[: -(len(track_id) + len('.mid'))]
			if track_id not in used:
				info.append((track_id, composer))
				used.append(track_id)
	with open("composers.txt", "w") as f:
		for track_id, composer in info:
			f.write(track_id + '\t' + composer + '\n')



if __name__ == '__main__':
	main()