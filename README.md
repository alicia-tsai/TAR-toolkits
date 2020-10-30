# TAR-toolkits: Text Analytics for Reconnaissance

This repository contains scripts that query earthquakes from USGS and collect data from twitter and news.

## Scripts and files
- `src/`: Source codes
- `data/data2template.py`: Convert earthquake log csv file from USGS to sentence description.
- `keys.ini`: Credentials file.
- `config.ini`: Default parameters.
- `data/earthquakes_log.csv`: All the earthquakes queried so far.
- `data/news/`: News data.
- `data/twitter/`: Twitter data.
- `notebooks/`: Jupyter notebook demo or work in progress results.

## Packages (Python 3):
- tweepy
- pandas
- newsapi
- calander
- pytz
- bs4
- datetime
- dateutil
- tzwhere
- csv
- re
- urllib
- selenium
- requests
- PIL
- PyPDF2
- textract
- nltk
- [cv2](https://pypi.org/project/opencv-python/): ```pip install opencv-python```
- pytesseract
- time

## Usage
Data collection can also be invoked manually by running `earthquakes_data.py`. This script will query earthquakes that happen today. To query from a specific date, change the variable `date` in the script.
```
python3 src/earthquakes_data.py
```

## Tools Installation

### NLTK
In order to split the sentences, package NLTK is used herein. The package needs to be installed first, then the following code should be invoked in Python to download the pretrained models:
```
nltk.download()
```
Note that the downloading process is only required in the first time.

### Selenium
In order to inspect pages, package ```selenium``` is needed, which could be downloaded [here](https://selenium-python.readthedocs.io/installation.html#downloading-python-bindings-for-selenium). I used Chrome version, so the package could automatically open Chrome. After downloading, run it, and fill in below the executable_path (the location of the executable file on the disk, e.g. in my case, it is /Users/lichenglong/Downloads/chromedriver) in the python codes when the package is used.

### Pytesseract
In order to extract text from images, Tesseract is used. The python package pytesseract is a python wrapper. Follow the following instructions to install and configure the tool.
- Tesseract needs to be installed by following the [link](https://tesseract-ocr.github.io/tessdoc/Home.html). For example, I am using MacOS, so I used ```homebrew```.
- pytesseract needs to be installed like normal packages (e.g. ```pip install pytesseract```)
- In Python, configure the path to Tesseract. In the official manual (in the link above) the path could be checked in terminal by following instructions (e.g., ```brew info tesseract``` for MacOS). HOWEVER, it does not work for my case. I found the path by using the following line in terminal: ```which tesseract```. The path could be configured by calling ```pytesseract.pytesseract.tesseract_cmd = YOUR PATH``` in python code.
    

