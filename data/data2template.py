# -*- coding: utf-8 -*-
import os
import re
import csv
import pandas
import calendar
import pytz
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import tz
from tzwhere import tzwhere

def process(source_data, dest_file, index):
    """
    source_data: csv data file, with directory to the csv file.
    dest_file: txt template.
    index: the index of event to be processed (starting at 0).
    """
    df = pd.read_csv("earthquakes_log.csv")   
    utctime = df.iloc[index]['rupture_time']
    longitude = df.iloc[index]['longitude']
    latitude = df.iloc[index]['latitude']

    # UTC to local time (daylight save is also considered)
    tzwherev = tzwhere.tzwhere()
    timezone_str = tzwherev.tzNameAt(latitude, longitude) # Seville coordinates
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz(timezone_str)
    utc = datetime.strptime(utctime, '%Y-%m-%d %H:%M:%S')
    utc = utc.replace(tzinfo=from_zone)
    localtime = utc.astimezone(to_zone)

    result = "On {} {}, {}, at approximately {}:{} local time, ".format(calendar.month_name[localtime.month], localtime.day, localtime.year, localtime.hour, localtime.minute); 
    result += "a magnitude {} earthquake, with a depth of {} km, ".format(df.iloc[index]['magnitude'], df.iloc[index]['depth']);
    location = df.iloc[index]['place'];
    location = re.split('km | | of', location);
    direction = {"E": "East", "W": "West", "S": "South", "N": "North", "SE": "Southeast", "SW": "Southwest", "NE": "Northeast", "NW": "Northwest", "ESE": "East-Southeast", "WSW": "West-Southwest", "ENE": "East-Northeast", "WNW": "West-Northwest", "SSE": "South-Southeast", "SSW": "South-Southwest", "NNE": "North-Northeast", "NNW": "North-Northwest"}
    city = "";
    for i in range(3, len(location)):
        city += location[i] + " ";
    city = city[:-1];
    result += "struck {} km {} of {}. ".format(location[0], direction[location[1]], city);
    if (float(latitude) > 0):
        latitudeNS = "N";
    else:
        latitudeNS = "S";
    if (float(longitude) > 0):
        longitudeEW = "E";
    else:
        longitudeEW = "W";
    result += "The coordinate of epicenter of the earthquake was {}°{}, {}°{}.".format(abs(float(latitude)), latitudeNS, abs(float(longitude)), longitudeEW);
    result += "\n";
    text = open("sample.txt", "a");
    text.write(result);
    text.close();

process("earthquakes_log.csv", "record.txt", 1);