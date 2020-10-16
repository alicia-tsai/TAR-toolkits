# Classification Task with Linear Models

This directory contains scripts and datasets for training and running classfication on earthquake reports.
The python files do cleaning, lemmatizing, vectorizing, training, and validating for the given csv data of earthquake reports containing three classes.

## File
- `classification_mj.ipynb` is a jupyter notebook containing examples for training and running classfication on earthquake reports.
- `classify_linear.py` is a python script that you can use to run on your choice of dataset/model.

## How to Run (`classify_linear.py`)
Simply run `python3 classify_linear.py datafile_name model_name`.
Here, the datafile must be a `.csv` file, and model_name is either "lr" (for logistic/softmax regression) or "svm" (for SVM model).


# Dependency
You must have following programs/libraries on your machine
- python3
- nltk
- numpy
- matplotlib
- sklearn
