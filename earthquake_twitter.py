import os
import requests
import tweepy as tw
import pandas as pd
from datetime import date


def get_tweet(keys, search_words, since_date=str(date.today()), max_item=500, lang="en", geocode=None):
    consumer_key, consumer_secret, access_token, access_token_secret = keys
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Define the search term and date
    search_words = search_words + " -filter:retweets"

    # Collect tweets
    tweets = tw.Cursor(api.search, q=search_words, lang=lang, geocode=geocode, since=since_date).items(max_item)

    users_locs = [[tweet.user.screen_name, tweet.user.location,
                   tweet.created_at, tweet.coordinates, tweet.text] for tweet in tweets]
    tweet_text = pd.DataFrame(data=users_locs, columns=['user', "location", "date", "coordinates", "content"])

    return tweet_text


def query_earthquake():
    # TODO: move auth key to a separate file
    consumer_key = 'vi360ysd3PGoNN0ZrpexlhMGi'
    consumer_secret = 'ZRQLsBhRa8FLfOT2OfpZoKZucHcEKvcj9OWrWIsDAa4i575T78'
    access_token = '1164687316713009152-KRzfNTvtWTwANW13TsHNv2wsLitbZZ'
    access_token_secret = 'YFwLyzwmIClv0VhOPko8i6Adss8macOzFOIBF8l6iKKSs'

    keys = [consumer_key, consumer_secret, access_token, access_token_secret]
    earthquake_tweet = {}

    #today = str(date.today())
    today = '2019-12-15'
    alertlevels = ["yellow", "orange", "red"]

    # Query earthquake with magnatuide >= 5 and alert level orange (yellow, orange, red)
    for alert in alertlevels:
        URL = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&minmagnitude=5&alertlevel={}&starttime={}".format(alert, today)
        response = requests.get(URL)
        earthquakes = response.json()['features']
        print("{0} {1} earthquake today.".format(len(earthquakes), alert))

        if len(earthquakes) > 0:
            for i in range(len(earthquakes)):
                location = earthquakes[i]["properties"]["place"].split()[-1] # get earthquake location
                search_words = location + "+earthquake"
                tweet_text = get_tweet(keys, search_words, since_date=today)
                earthquake_tweet[location] = tweet_text

    # Store tweet to csv file
    for location in earthquake_tweet.keys():
        earthquake_tweet[location].to_csv(os.path.join("data", "twitter", "{}_{}_earthquake.csv".format(today, location)), index=False)


if __name__ == "__main__":
    query_earthquake()
