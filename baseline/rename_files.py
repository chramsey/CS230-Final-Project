import os

"""
Used for preprocessing:
Renames each file in terms of its composer (which will be used as its final label) and global id.
"""

dataset = "NN_midi_files_extended"

def main():
	track_id = 0
	for dir_name, subdir_list, file_list in os.walk(dataset):
		composer = dir_name[len(dataset):]
		file_path_list = ['/'.join([dir_name, file]) for file in file_list if '.mid' in file]
		for file_path in file_list:
			if '.mid' not in file_path: 
				continue
			track_id_str = "{0:0=3d}".format(track_id)
			os.rename(dir_name + '/' + file_path, dir_name + '/' + composer + track_id_str + '.mid')
			track_id += 1



if __name__ == '__main__':
	main()
