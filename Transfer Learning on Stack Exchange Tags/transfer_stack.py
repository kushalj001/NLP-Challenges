import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import nltk
import spacy
import re
import string
import gc
import gensim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Embedding,Conv1D,Dropout,LSTM,MaxPool1D
from keras.models import Sequential
from keras.models import load_model


robotics_data = pd.read_csv('transfer-learning-on-stack-exchange-tags/robotics.csv')
diy_data = pd.read_csv('transfer-learning-on-stack-exchange-tags/diy.csv')
biology_data = pd.read_csv('transfer-learning-on-stack-exchange-tags/biology.csv')
crypto_data = pd.read_csv('transfer-learning-on-stack-exchange-tags/crypto.csv')
travel_data = pd.read_csv('transfer-learning-on-stack-exchange-tags/travel.csv')
cooking_data = pd.read_csv('transfer-learning-on-stack-exchange-tags/cooking.csv')

data = pd.concat([robotics_data,diy_data,biology_data,travel_data,cooking_data,crypto_data])

def remove_html_tags(html):
    soup = BeautifulSoup(html,'lxml')
    text = soup.get_text()
    return text

stopwords = nltk.corpus.stopwords.words('english')
#print(stopwords)
def remove_urls(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    return text

def remove_punctuation(text):
    chars = [char for char in text if char not in string.punctuation]
    text = ''.join([char for char in chars])
    return text

def remove_stopwords(text):
    text_lower = [x.lower() for x in text]
    text = ''.join([x for x in text_lower])
    tokens = nltk.word_tokenize(text)
    processed_text = [word for word in tokens if word not in stopwords]
    processed_text = ' '.join([word for word in processed_text])
    return processed_text

def lemmatize_text(text):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    clean_text = ' '.join([word for word in lemmas])
    return clean_text

data['content'] = data['content'].apply(lambda x : remove_html_tags(x))
data['title'] = data['title'].apply(lambda x: remove_urls(x))
data['title'] = data['title'].apply(lambda x: remove_punctuation(x))
data['title'] = data['title'].apply(lambda x: remove_stopwords(x))
data['title'] = data['title'].apply(lambda x: lemmatize_text(x))

data['content'] = data['content'].apply(lambda x: remove_urls(x))
data['content'] = data['content'].apply(lambda x: remove_punctuation(x))
data['content'] = data['content'].apply(lambda x: remove_stopwords(x))
data['content'] = data['content'].apply(lambda x: lemmatize_text(x))

data['tags'] = data['tags'].apply(lambda x:x.split())

embeddings = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True,limit=600000)


max_words = 50000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(data['content']))
sequences = tokenizer.texts_to_sequences(list(data['content']))
data_set = pad_sequences(sequences,maxlen=max_len)

one_hot = preprocessing.MultiLabelBinarizer()
one_hot_labels = one_hot.fit_transform(data['tags'])
#print(one_hot_labels.shape)

train_X = data_set[:int(-0.3*data_set.shape[0])]
val_X = data_set[int(-0.3*data_set.shape[0]):]
train_y = one_hot_labels[:int(-0.3*data_set.shape[0])]
val_y = one_hot_labels[int(-0.3*data_set.shape[0]):]

embeddings_index = {}
embedding_size = 300
for word in embeddings.wv.vocab:
    embeddings_index[word] = embeddings.word_vec(word)

all_embeddings = np.stack(list(embeddings_index.values()))
embed_mean,embed_std = all_embeddings.mean(),all_embeddings.std()
num_words = len(tokenizer.word_index)

embedding_matrix = np.random.normal(embed_mean,embed_std,(num_words,embedding_size))

for word,index in tokenizer.word_index.items():
    index -= 1
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

del(embeddings_index)
gc.collect()

embedding_layer = Embedding(len(tokenizer.word_index),300,weights = [embedding_matrix],trainable=False)
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=512,kernel_size=7,activation='relu'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(0.3))
model.add(Conv1D(filters=512,kernel_size=7,activation='relu'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(one_hot.classes_), activation='relu'))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(train_X,train_y,validation_data=(val_X,val_y),epochs=3,batch_size=256,verbose=1)

model.save('tags_model.h5')
model = load_model('tags_model.h5')

test_data = pd.read_csv('transfer-learning-on-stack-exchange-tags/test.csv')
test_data['content'] = test_data['content'].apply(lambda x:remove_html_tags(x))
test_data['content'] = test_data['content'].apply(lambda x:remove_punctuation(x))
test_data['content'] = test_data['content'].apply(lambda x:remove_stopwords(x))
test_data['content'] = test_data['content'].apply(lambda x:lemmatize_text(x))

test_X = pad_sequences(tokenizer.texts_to_sequences(list(test_data['content'])),maxlen=max_len)

predictions = model.predict(test_X,batch_size=256,verbose=1)
