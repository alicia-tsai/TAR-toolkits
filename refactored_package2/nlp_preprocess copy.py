import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import nltk
max_sentence_len = 64
    # Minjune's cleaning functions
def clean(txt_lst):   
   # print(txt_lst)
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
        tokens = nltk.WordPunctTokenizer().tokenize(text)
        if(len(tokens) > 0): 
            tokens = np.pad(tokens, (0, max_sentence_len - len(tokens)), 'constant', constant_values=(None, ''))
        return tokens
    return list(map(clean_text, txt_lst))

def lemmatize(txt_lst):
    lemm = nltk.stem.WordNetLemmatizer()
    return list(map(lambda word: list(map(lemm.lemmatize, word)),
                    txt_lst))
# Get vocabulary: {word1: 1, word2: 2, word3: 3} of word and int pairs in a dictionary for embedding purposes
def get_vocab(column):
    corpus = []
    for i in range(len(column)):
        tokens = column.iloc[i] # Token array of words might need to refine
        assert type(tokens) == list
       # print(tokens)
        corpus = np.concatenate((corpus, tokens)) # add individual words into corpus array
    vocab = {k: v for v, k in enumerate(np.unique(corpus))}
    return vocab

# Using the vocab dictionary to translate list of senteces into list of integer arrays
# return x_by_class and y_by_class
def word_to_ints(df, vocab, num_classes):
    x_by_class = [[] for i in range(num_classes)]
    y_by_class = [[] for i in range(num_classes)]
    for i in range(len(df)):
        tokens = df.iloc[i]['lemmatized'] # List of strings
        int_tokens = np.array([], int) # initilaize an empty list of integer arrays
        for t in tokens:
            if t in vocab:
                int_tokens = np.append(int_tokens, vocab[t])
            else:
                int_tokens = np.append(int_tokens, 0)
        # Then figure out which class to put this list of int into: Ignore the "Other" class for now
        if "building" in df.iloc[i]['label'].lower():
            x_by_class[0].append(int_tokens)
            y_by_class[0].append([1.0, 0.0, 0.0])
        elif "infrastructure" in df.iloc[i]['label'].lower():
            x_by_class[1].append(int_tokens)
            y_by_class[1].append([0.0, 1.0, 0.0])
        elif "resilience" in df.iloc[i]['label'].lower():
            x_by_class[2].append(int_tokens)
            y_by_class[2].append([0.0, 0.0, 1.0])
    
    # convert to numpy array
    for i in range(num_classes):
        x_by_class[i] = np.array(x_by_class[i])
        y_by_class[i] = np.array(y_by_class[i])
        
    return x_by_class, y_by_class       

def word_to_ints_concat_class(df, vocab):
    x = []
    y = []
    for i in range(len(df)):
        tokens = df.iloc[i]['lemmatized'] # List of strings
        int_tokens = np.array([], int) # initilaize an empty list of integer arrays
        for t in tokens:
            if t in vocab:
                int_tokens = np.append(int_tokens, vocab[t])
            else:
                int_tokens = np.append(int_tokens, 0)
                
        # Then figure out which class to put this list of int into: Ignore the "Other" class for now
        if "building" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([1.0, 0.0, 0.0])
        elif "infrastructure" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([0.0, 1.0, 0.0])
        elif "resilience" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([0.0, 0.0, 1.0])
    
    return x, y

def word_to_ints_concat_class_raw(df, vocab):
    x = []
    y = []
    for i in range(len(df)):
        tokens = df.iloc[i]['lemmatized'] # List of strings
        int_tokens = np.array([], int) # initilaize an empty list of integer arrays
        for t in tokens:
            if t in vocab:
                int_tokens = np.append(int_tokens, vocab[t])
            else:
                int_tokens = np.append(int_tokens, 0)
                
        # Then figure out which class to put this list of int into: Ignore the "Other" class for now
        if "building" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([1.0, 0.0, 0.0, 0.0])
        elif "infrastructure" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([0.0, 1.0, 0.0, 0.0])
        elif "resilience" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([0.0, 0.0, 1.0, 0.0])
        else:
            x.append(int_tokens)
            y.append([0.0, 0.0, 0.0, 1.0])
    
    return x, y

def preprocess_data(train, test): 
    df_train = pd.read_csv(train)  
    df_test = pd.read_csv(test)
    
    
    # Count the length of the longest sentences in each dataset
 #   df_test.rename(columns = {"Sentence", })
    
    
    

    max_length_train = max([len(df_train.sentence.iloc[i].split(" ")) for i in range(len(df_train))])
    max_length_test = max([len(df_test.sentence.iloc[i].split(" ")) for i in range(len(df_test))])
    print('max train sentence len '+str(max_length_train))
    print('max test sentence len '+str(max_length_test))
    max_sentence_len = 64

    # Clean sentences
    df_train['cleaned'] = clean(df_train['sentence'])
    df_train['lemmatized'] = lemmatize(df_train['cleaned'])

    df_test['cleaned'] = clean(df_test['sentence'])
    df_test['lemmatized'] = lemmatize(df_test['cleaned'])
# Note, cleaned and lemmatized columns now contain lists of words
    vocab = get_vocab(df_train['lemmatized'])

    vocab_train_size = len(vocab)
    
    x_train_by_class, y_train_by_class = word_to_ints(df_train, vocab, 3) 
    x_test_by_class, y_test_by_class = word_to_ints(df_test, vocab, 3)

    file = open("numpyData/x_test_by_class", "w") 
    file = open("numpyData/t_test_by_class", "w") 
    file = open("numpyData/x_train_by_class", "w") 
    file = open("numpyData/y_train_by_class", "w") 
    
    
    file = open("numpyData/x_train", "w") 
    file = open("numpyData/y_train", "w") 
    
    file = open("numpyData/x_test", "w") 
    file = open("numpyData/y_test", "w") 
        
    file = open("numpyData/raw_x_test", "w") 
    file = open("numpyData/raw_y_test", "w") 
    
   # file = open("nump")
    np.save('numpyData/x_test_by_class', x_test_by_class)
    np.save('numpyData/y_test_by_class', y_test_by_class)
    np.save('numpyData/x_train_by_class', x_train_by_class)
    np.save('numpyData/y_train_by_class', y_train_by_class)

    x_train, y_train = word_to_ints_concat_class(df_train, vocab)
    x_test, y_test = word_to_ints_concat_class(df_test, vocab)
    
    np.save('numpyData/x_train', x_train)
    np.save('numpyData/y_train', y_train)
    np.save('numpyData/x_test', x_test)
    np.save('numpyData/y_test', y_test)
    
    raw_x_test, raw_y_test = word_to_ints_concat_class_raw(df_test, vocab)
    np.save('numpyData/raw_x_test', raw_x_test)
    np.save('numpyData/raw_y_test', raw_y_test)
    
def word_to_ints_concat_class_raw(df, vocab):
    x = []
    y = []
    
    for i in range(len(df)):
        tokens = df.iloc[i]['lemmatized'] # List of strings
        int_tokens = np.array([], int) # initilaize an empty list of integer arrays
        for t in tokens:
            if t in vocab:
                int_tokens = np.append(int_tokens, vocab[t])
            else:
                int_tokens = np.append(int_tokens, 0)
                
        # Then figure out which class to put this list of int into: Ignore the "Other" class for now
        if "building" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([1.0, 0.0, 0.0, 0.0])
        elif "infrastructure" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([0.0, 1.0, 0.0, 0.0])
        elif "resilience" in df.iloc[i]['label'].lower():
            x.append(int_tokens)
            y.append([0.0, 0.0, 1.0, 0.0])
        else:
            x.append(int_tokens)
            y.append([0.0, 0.0, 0.0, 1.0])
    
    return x, y


