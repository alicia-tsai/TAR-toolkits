from datetime import datetime
from collections import defaultdict 

from matplotlib import pyplot as plt
import math

data = pd.read_csv("dataalbania.csv") 
chosen = ((data["date"][0]))

def convertToTime(inputString): 
    year = int(inputString[0:4])
    month = int(inputString[5:7])
    day = int(inputString[8:10])
    hour = int(inputString[11:12 + 1])
    minute = int(inputString[14:16])
    seconds = int(inputString[17:19])

    dateObject = datetime(year, month, day, hour, minute, seconds) 
    
    return dateObject

def dateDifference(date1, date2): 
    return divmod((convertToTime(date1) - convertToTime(date2)).total_seconds(), 3600 * 24)

earthquakeLog = pd.read_csv("data/earthquakes_log.csv") 
twitterCSV = pd.read_csv("data/twitter/tweets.csv") 

def generateResilience(rupture_time, twitterFile, keywords): 
    frequencies = defaultdict(int) 

    tweets = pd.read_csv(twitterFile)

    tweets = tweets.to_numpy() 


    for x in range(0, len(tweets)): 
        if(any([keywords[j] in tweets[x][4] for j in range(0, len(keywords))])):
            differenceTweet = (dateDifference(tweets[x][2], ruptureTime)[0]) 
            frequencies[differenceTweet] += 1

            
    t0Group = (list(filter(lambda x: x[1] == max(frequencies.values()), frequencies.items())))[0]
    
    reversedDictionary = (list(frequencies.items())[::-1])

    t1 = 0 
    
    for x in range(0, len(reversedDictionary)): 
        threshold = 0.1 * t0Group[1] 
    
        if(reversedDictionary[x][1] > 0.1 * t0Group[1]): 
            valueBefore = reversedDictionary[x - 1][1] 
            valueNow = reversedDictionary[x][1]
        
            valueDifference = valueNow - valueBefore
            
            if(valueDifference == 0): 
                t1 = reversedDictionary[x - 1][0]
            else: 
                t1 = reversedDictionary[x - 1][0] - (threshold - valueBefore) / valueDifference
            
    x = frequencies.keys() 
    y = frequencies.values()

    
    plt.plot(x, y)
        
    plt.show()
    
    retList = [t0Group[0], t1] 
    
    return retList
    
generateResilience("2021-02-24 02:05:59", "data/twitter/tweets.csv", "Iceland", ["earthquake", "electricity"])