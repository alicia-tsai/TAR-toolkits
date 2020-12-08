# TAR Package

## Files
- `tar_main.py`: Primary file to run different parts of the code
- `classifiers.py`: File that aggregates the 4 classifiers into 1 output CSV 
- `summarize.py`: Summarizes each category and generates Word document 
- `gan_model.py`: Billy's GAN Model 
- `cnn_model.py`: Billy's CNN Model 
- `classify_linear.py`: Minjune's Classifiers 
- `nlp_Preprocess.py`: Billy's preprocessing file

## Usage

To run, call `tar_main.py`, with 2 command line arguments: the training dataset, and the testing dataset. 

Here we have `dataset.csv` as the training set `Albania.csv` as test set. 

```
python3 tar_main.py dataset.csv Albania.csv
```