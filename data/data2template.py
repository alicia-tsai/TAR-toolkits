# -*- coding: utf-8 -*-
import os
import re
import csv
import calendar

def process(source_data, dest_file, index):
    """
    source_data: csv data file, with directory to the csv file.
    dest_file: txt template.
    index: the index of event to be processed (starting at 1).
    """
    with open("earthquakes_log.csv") as f:
        reader = csv.reader(f)
        mylist = list(reader)    
    time = str(mylist[index][mylist[0].index('rupture_time')])
    time = re.split('-| |:', time)
    result = "On {} {}, {}, at approximately {}:{} UTC, ".format(calendar.month_name[int(time[1])], time[2], time[0], time[3], time[4]); 
    result += "a magnitude {} earthquake, with a depth of {} km, ".format(mylist[index][mylist[0].index('magnitude')], mylist[index][mylist[0].index('depth')]);
    location = mylist[index][mylist[0].index('place')];
    location = re.split('km | | of', location);
    direction = {"E": "East", "W": "West", "S": "South", "N": "North", "SE": "Southeast", "SW": "Southwest", "NE": "Northeast", "NW": "Northwest", "ESE": "East-Southeast", "WSW": "West-Southwest", "ENE": "East-Northeast", "WNW": "West-Northwest", "SSE": "South-Southeast", "SSW": "South-Southwest", "NNE": "North-Northeast", "NNW": "North-Northwest"}
    city = "";
    for i in range(3, len(location)):
        city += location[i] + " ";
    city = city[:-1];
    result += "struck {} km {} of {}. ".format(location[0], direction[location[1]], city);
    if (float(mylist[index][mylist[0].index('latitude')]) > 0):
        latitude = "N";
    else:
        latitude = "S";
    if (float(mylist[index][mylist[0].index('longitude')]) > 0):
        longitude = "E";
    else:
        longitude = "W";
    result += "The coordinate of epicenter of the earthquake was {}°{}, {}°{}.".format(abs(float(mylist[index][mylist[0].index('latitude')])), latitude, abs(float(mylist[index][mylist[0].index('longitude')])), longitude);
    result += "\n";
    text = open("sample.txt", "a");
    text.write(result);
    text.close();

process("earthquakes_log.csv", "record.txt", 1);