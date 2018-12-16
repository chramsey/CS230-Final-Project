place all midi files in data/sets
 - this includes the training, validation, and test set, which will be randomly allocated during data loading

choose params in utils.py:
 - piano roll size (how many sample per input)
 - window size
 - fs (number of samples per second) 

run github.py 
 - this will load the data with those parameters
 - you will be prompted for values for hyperparameters before a model is trained. 
 - values will print and be logged to file (output_date_time)
 - continue changing hyperparams and running models as desired

repeated with different values for params in utils.py
