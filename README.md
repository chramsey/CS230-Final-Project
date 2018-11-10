# Classifying Musical Scores by Composer
## CS230 Final Project

## Authors
Chenduo Huang || cdhuang@stanford.edu

Christina Ramsey || cmramsey@stanford.edu

Daniel Costa || dcosta2@stanford.edu

## Introduction
We  are  interested  in  classifying  musical  scoresby composer.  As this is an ambitious task, wedecided to deal with specific classical composersin  order  to  limit  the  scope.   That  being  said,we have enough data to expand into additionaleras, genres, or composers if desired.  We foreseethat this project will be an interesting experi-ment in audio classification and will potentiallydemonstrate  where  composers  were  influencedby other composers.

## Dataset
The dataset we have been using for our classification so far is found in NN_midi_files. This dataset consists of 200 songs from 4 different composers. NN_midi_files extended consists of 450 songs from 9 composers, and will likely be the dataset we use moving forward.

## Baseline
Files relating to our baseline can be found in the baseline folder, which consists of data_preprocessing (renaming files and creating a mapping of track_ids to composer labels) and a classifier which achieves an accuracy of 0.71875 on the NN_midi_files dataset.
