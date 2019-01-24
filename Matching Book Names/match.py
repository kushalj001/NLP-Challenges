import imp
import numpy as np
import pandas as pd
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

n = int(input())
book_names = [str(input()) for i in range(n)]
delimiter =  input()
descriptions = [str(input()) for i in range(n)]

descriptions_df = pd.DataFrame(descriptions,columns=['desc'])


# Downloading the nltk stopwords takes time and results in timeout.
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',
  'them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these',
  'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
  'does', 'did', 'doing' ,'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under' ,'again', 'further',
  'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
  'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
  'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
  "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
  'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def preprocess(text):
    '''Combination of all the necassary text preprocessing'''
    tokens = re.split('\W+',text)
    tokens = [word.lower() for word in tokens]

    return text

# Fitting a TF-IDF vectorizer on description docs
descriptions_df['cleaned'] = descriptions_df['desc'].apply(lambda x:preprocess(x))
vectorizer = TfidfVectorizer(stop_words = stopwords)
features = vectorizer.fit_transform(descriptions_df['cleaned'])

# Transforming book names into feature vectors according to the vocabulary learnt based on dictionaries
book_names_df = pd.DataFrame(book_names,columns=['book names'])
name_features = vectorizer.transform(book_names_df['book names'])

# Converting features into arrays
desc_vectors = features.toarray()
book_vectors = name_features.toarray()

# Dot product to calculate the closest matching book for each description
x = np.dot(book_vectors,desc_vectors.T)
answers = list(np.argmax(x,axis=0))

# Print answers
for answer in answers:
    print(answer+1)
