import matplotlib.pyplot as plt
import sklearn
import nltk
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import sys

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix

def clean(txt_lst):   
    def clean_text(text, remove_stopwords = True):
        text = text.lower()
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text) 
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)
        return nltk.WordPunctTokenizer().tokenize(text)
    return list(map(clean_text, txt_lst))

def lemmatize(txt_lst):
    lemm = nltk.stem.WordNetLemmatizer()
    return list(map(lambda word: list(map(lemm.lemmatize, word)),
                    txt_lst))


if __name__ == '__main__':

	datafile = sys.argv[0]
	mode = sys.argv[1]

	df = pd.read_csv(datafile)

	train_data, validate_data = sklearn.model_selection.train_test_split(df, train_size = 0.65, random_state=42)

	df['cleaned'] = clean(df['sentence'])
	df['lemmatized'] = lemmatize(df['cleaned'])  

	bow_converter = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)

	x_train = bow_converter.fit_transform(train_data['lemmatized'])
	x_validate = bow_converter.transform(validate_data['lemmatized'])

	y_train = train_data["label"]
	y_validate = validate_data["label"]

	if mode == "lr":
		model = LogisticRegression(C=1)
	else if mode == "svm"
		model = svm.LinearSVC()

	model.fit(x_train, y_train)
	train_score = model.score(x_train, y_train)
	test_score = model.score(x_validate, y_validate)
	print('Train Score: ', train_score)
	print('Test Score: ', test_score)
