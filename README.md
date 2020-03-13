# TAR-toolkits: Text Analytics for Reconnaissance

This repository contains scripts that query earthquakes from USGS and collect data from twitter and news.

## Scripts and files
- `src/earthquakes_data.py`: Query earthquakes with magnitude >= 5 and alert levels in yellow, orange, or red. Collect and store twitter and news data.
- `src/scheduler.py`: Schedule automatic data collection every day. This file is to be run in background.
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

## Usage
1. For automatic data collection, run `scheduler.py` in background. It is scheduled to query USGS every day at 1 am.
```
python3 scheduler.py
```

2. Data collection can also be invoked manually by running `earthquakes_data.py`. This script will query earthquakes that happen today. To query from a specific date, change the variable `date` in the script.
```
python3 earthquakes_data.py
```
