###load midi files and modify it here


import os



def parse_data():
    midi_data = pretty_midi.PrettyMIDI('NN_midi_files/byrd/byrd50.mid')

    all_data = []
    cutoff = 10000

    this_dir_name = '/Users/dcosta/NN_proj/NN_midi_files'
    boole = True
    directory = os.fsencode(this_dir_name)
    i = 1
    for dirnm in os.listdir(directory):
        dirname = str(os.fsdecode(dirnm))
        if dirname[0] == '.' or dirname == 'train':
            continue
        new_dir = str(directory)[2:-1] + '/' + dirname
        for file in os.listdir(new_dir):
            filename = os.fsdecode(file)
            if filename[0] == '.':
                continue
            with open(new_dir + '/' + filename, 'rb') as b50:
                composer = dirname
                track_id = i
                os.rename(str(new_dir) + '/' + str(filename), str(new_dir) + '/' + str(composer) + "{0:0=3d}".format(track_id) + ".mid")
                content = b50.read()
                #print("Filename = " + filename + " - Filesize = " +  str(len(list(content))))
                example_n = list(content)
                example_n_sz = len(example_n)
                if example_n_sz > cutoff:
                    example_n = example_n[:10000]
                if example_n_sz < cutoff:
                    example_n = example_n + [0 for i in range(cutoff - example_n_sz)]
                all_data.append((example_n, composer))
                i += 1
    return all_data
