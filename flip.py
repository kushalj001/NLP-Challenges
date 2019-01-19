import numpy as np
import pandas as pd
import nltk
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

data = pd.read_csv('./train.tsv',sep='\t',names=['product','label'])
X = data['product']
y = data['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


stopwords = nltk.corpus.stopwords.words('english')
def remove_punctuation(text):
    '''Removes punctuation from the corpus as they do not contribute any information.'''
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def tokenize(text):
    '''A crude form of tokenization by splitting the sentences with whitespace.'''
    tokens = re.split('\W+',text)
    return tokens

def remove_stopwords(text):
    '''Removes common words in english that occur too frequently in our text and do not add any valuable info to model.'''
    text = [word.lower() for word in text]
    clean_text = [word for word in text if word not in stopwords]
    return clean_text

def preprocess(text):
    '''Combination of all the necassary text preprocessing'''
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = re.split('\W+',text)
    tokens = [word.lower() for word in tokens]
    text = " ".join([word for word in tokens if word not in stopwords])

    return text


# Tf-idf vectorizer. Converts our tokens into numeric features. The vectorizer object is initialized with an analyzer which
# is the preprocess function

vectorizer = TfidfVectorizer(analyzer=preprocess)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_train_tfidf.shape

# Support Vector Machine Classifier.SVM params found using grid search.
svm = SVC(C=10,gamma=0.1)
svm.fit(X_train_tfidf,y_train)

# A pipeline for NLP
svm_pipeline = Pipeline([('tfidf',TfidfVectorizer()),('svm',LinearSVC())])
svm_pipeline.fit(X_train,y_train)

# Evaluation
predictions = svm_pipeline.predict(X_test)
#print(metrics.accuracy_score(y_test,predictions))
#print(metrics.classification_report(y_test,predictions))

n = int(input())
text = [input() for _ in range(n)]
features = vectorizer.transform(text)
predictions = svm_pipeline.predict(features)

for pred in predictions:
    print(pred)
