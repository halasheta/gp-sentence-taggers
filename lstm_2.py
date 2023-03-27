#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import brown
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import brown, treebank
from tensorflow.keras.optimizers import Adam
from gensim.models import KeyedVectors
import gensim.downloader as api
from nltk.corpus import conll2000
import os

path = api.load("word2vec-google-news-300", return_path=True)
tf.config.run_functions_eagerly(True)

# In[2]:


nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
nltk.download('treebank')
nltk.download('conll2000')

brown_sent = brown.tagged_sents(tagset='universal')
tree_sent = treebank.tagged_sents(tagset='universal')
conll_sent = conll2000.tagged_sents(tagset='universal')
all_sent = brown_sent + tree_sent + conll_sent

# In[22]:


# Tokenize the words

word = [[word[0].lower() for word in tup] for tup in all_sent]  # store the word
pos = [[pos[1] for pos in tup] for tup in all_sent]  # store the corresponding pos tag

word_tokenizer = Tokenizer()
pos_tokenizer = Tokenizer()

word_tokenizer.fit_on_texts(word)
word_seqs = word_tokenizer.texts_to_sequences(word)
pos_tokenizer.fit_on_texts(pos)
pos_seqs = pos_tokenizer.texts_to_sequences(pos)

# In[24]:


# find the length of the training sets
max_len = 100  # don't need to pad fixed len dataset
w_size = len(word_tokenizer.word_index) + 1
pos_size = len(pos_tokenizer.word_index) + 1

# In[25]:


# code modified from https://towardsdatascience.com/pos-tagging-using-rnn-7f08a522f849
word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
embedding_size = 300
embedding_weights = np.zeros((w_size, embedding_size))
word2id = word_tokenizer.word_index
for word, index in word2id.items():
    try:
        embedding_weights[index, :] = word_vectors[word]
    except KeyError:
        pass

# the LSTM Model
model = Sequential()
optimizer = Adam(learning_rate=0.001)

model.add(Embedding(w_size, embedding_size, input_length=max_len, weights=[embedding_weights], trainable=False))
model.add(LSTM(64, return_sequences=True))
model.add(TimeDistributed(Dense(pos_size, activation='softmax')))
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

words_padded = pad_sequences(word_seqs, max_len, padding='post', truncating='post')
pos_padded = pad_sequences(pos_seqs, max_len, padding='post', truncating='post')

pos_padded = to_categorical(pos_padded)

word_train, word_test, pos_train, pos_test = train_test_split(words_padded, pos_padded, test_size=0.2)  # 90 10
word_test, word_valid, pos_test, pos_valid = train_test_split(word_train, pos_train, test_size=0.5) #  80 10


training = model.fit(word_train, pos_train, batch_size=128, epochs=10, validation_data=(word_valid, pos_valid))
model.save('lstm_lr0.001_bs128_e10.h5')
loss, accuracy = model.evaluate(word_test, pos_test, verbose=1)
print('Model Done')
