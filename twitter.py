# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:23:36 2019

@author: Jing LIAN
"""

import os
import tweepy as tw
import pandas as pd
 
consumer_key = 'vi360ysd3PGoNN0ZrpexlhMGi'
consumer_secret = 'ZRQLsBhRa8FLfOT2OfpZoKZucHcEKvcj9OWrWIsDAa4i575T78'
access_token = '1164687316713009152-KRzfNTvtWTwANW13TsHNv2wsLitbZZ'
access_token_secret = 'YFwLyzwmIClv0VhOPko8i6Adss8macOzFOIBF8l6iKKSs'
 
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
search_words = "earthquake"
date_since = "2019-11-06"

new_search = "California+power+outage -filter:retweets"
new_search
# Collect tweets
tweets = tw.Cursor(api.search,
                       q=new_search,
                       lang="en",
                       #geocode="37.914753,-122.299857,20km",
                       since=date_since).items(300)

users_locs = [[tweet.user.screen_name, tweet.user.location,tweet.created_at,tweet.coordinates,tweet.text] for tweet in tweets]
tweet_text = pd.DataFrame(data=users_locs, columns=['user', "location","date","coordinates","content"])
tweet_text.to_csv('D:\\E\\博四上\\twitterdata_California_Earthquake_100km.csv')