import os
import requests
import tweepy as tw
import pandas as pd
from datetime import date
import configparser
from newsapi import NewsApiClient


def get_usgs(alert, date):
    URL = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&minmagnitude=5&alertlevel={}&starttime={}".format(alert, date)
    response = requests.get(URL)
    earthquakes = response.json()['features']
    print("{0} {1} earthquake today.".format(len(earthquakes), alert))
    return earthquakes


def get_tweet(keys, search_words, date=str(date.today()), max_item=500, lang="en", geocode=None):
    consumer_key, consumer_secret, access_token, access_token_secret = keys
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Define the search term and date
    search_words = search_words + " -filter:retweets"

    # Collect tweets
    tweets = tw.Cursor(api.search, q=search_words, lang=lang, geocode=geocode, since=date).items(max_item)

    users_locs = [[tweet.user.screen_name, tweet.user.location,
                   tweet.created_at, tweet.coordinates, tweet.text] for tweet in tweets]
    tweet_text = pd.DataFrame(data=users_locs, columns=['user', "location", "date", "coordinates", "content"])

    return tweet_text


def get_news(api_key, news_search_words, date=str(date.today()), page=2, lang="en"):
    newsapi = NewsApiClient(api_key=api_key)

    # endpoint: /v2/everything
    all_articles = newsapi.get_everything(q=news_search_words,
                                          from_param=date,
                                          language=lang,
                                          sort_by='relevancy',
                                          page=page)  # each page is 20 articles

    keywords = set(["quake", "earthquake", "tremors"])
    articles = []

    for article in all_articles['articles']:
        if keywords.intersection(article['title'].split()):
            articles.append([article['source']['name'], article['title'], article['url'],
                            article['publishedAt'][:10], article['content']])

    news = pd.DataFrame(data=articles, columns=['source_name', "title", "url", "date", "content"])

    return news

def get_earthquake_data(twitter_keys=None, news_key=None, date):
    earthquake_tweet = {}
    earthquake_news = {}
    alertlevels = ["yellow", "orange", "red"]

    # Query earthquake with magnitude >= 5 and alert level yellow, orange, red
    for alert in alertlevels:
        earthquakes = get_usgs(alert, date)

        if len(earthquakes) > 0:
            for i in range(len(earthquakes)):
                location = earthquakes[i]["properties"]["place"].split(", ")[-1] # get earthquake location
                print("  - {0}".format(location))
                twitter_search_words = location + "+earthquake"
                tweet_df = get_tweet(twitter_keys, twitter_search_words, date)
                earthquake_tweet[location] = tweet_df

                news_search_words = '+("earthquake" AND "{}")'.format(location)
                news_df = get_news(news_key, news_search_words, date)
                earthquake_news[location] = news_df

    # Store tweet to csv file
    for location in earthquake_tweet.keys():
        earthquake_tweet[location].to_csv(os.path.join("data", "twitter", "{}_{}_earthquake.csv".format(date, location)), index=False)
        earthquake_news[location].to_csv(os.path.join("data", "news", "{}_{}_earthquake.csv".format(date, location)), index=False)


if __name__ == "__main__":
    # Read in tweepy api and news api keys
    keysparser = configparser.ConfigParser()
    keysparser.read('keys.ini')
    twitter_keys = [keysparser['tweepyapi_key']['consumer_key'],
                    keysparser['tweepyapi_key']['consumer_secret'],
                    keysparser['tweepyapi_key']['access_token'],
                    keysparser['tweepyapi_key']['access_token_secret']
                    ]
    news_key = keysparser['newsapi_key']['api_key']

    # Query USGS and get social media and news data
    date = str(date.today()) # to query from a specific date, change this variable to something like '2020-01-07'
    get_earthquake_data(twitter_keys, news_key, date)
