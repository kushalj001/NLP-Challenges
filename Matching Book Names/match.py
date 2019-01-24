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

#desc_sentences = nltk.sent_tokenize(descriptions)
descriptions_df = pd.DataFrame(descriptions,columns=['desc'])

stopwords = nltk.corpus.stopwords.words('english')

def preprocess(text):
    '''Combination of all the necassary text preprocessing'''
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = re.split('\W+',text)
    tokens = [word.lower() for word in tokens]
    text = " ".join([word for word in tokens if word not in stopwords])

    return text

descriptions_df['cleaned'] = descriptions_df['desc'].apply(lambda x:preprocess(x))
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(descriptions_df['cleaned'])

#book_names = book_names.split('\n')
book_names_df = pd.DataFrame(book_names,columns=['book names'])
name_features = vectorizer.transform(book_names_df['book names'])

desc_vectors = features.toarray()
book_vectors = name_features.toarray()
x = np.dot(book_vectors,desc_vectors.T)
answers = list(np.argmax(x,axis=0))
for answer in answers:
    print(answer+1)
