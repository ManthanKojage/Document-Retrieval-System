#!/usr/bin/env python
# coding: utf-8

# In[15]:


# importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import glob
import re
import os
import numpy as np
import sys
import math
import pickle
import string
from num2words import num2words


# In[16]:


# functions for preprocessing of the documents in the courpus

# stemmming function uses porter stemmer to stemm the words and it return those words again
# by combining them to form strings
def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


# convert_numbers converts numeric data to alphabetic data by using num2words library
# this also returns the string formed by combining the words


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            pass
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


# remove_stop_words removes all the stopwords from the english dictionary
def remove_stop_words(data):
    # stop_words = stopwords.words('english')

    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


# docidx is a dictionaary consting values as file indexes(start from 0)

# and keys as the document name in the corpus

docidx = {}
file_folder = "english-corpora\\*"
idx = 0
for file in glob.glob(file_folder):
    docidx[os.path.splitext(os.path.basename(file))[0]] = idx
    idx = idx + 1

with open("Question1_output/dictionary_key_file_val_index", "wb") as handle:
    pickle.dump(docidx, handle, protocol=pickle.HIGHEST_PROTOCOL)


# doc_idx is a dictionaary consting keys as file indexes(start from 0)

# and value as the document name in the corpus


doc_idx = {}
file_folder = "english-corpora\\*"
idx = 0
for file in glob.glob(file_folder):
    doc_idx[idx] = os.path.splitext(os.path.basename(file))[0]
    idx = idx + 1

with open("Question1_output/dictionary_key_index_val_file", "wb") as handle:
    pickle.dump(doc_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


# In[17]:


# importing time to track the time in which the preprocessing is done(40mins approx)
import time

curr = time.time()
# stopwords of english dictioanry
stop_words = stopwords.words("english")
# all the punctuations
symbols = '!"#$%&()*+-./:;<=>,?@[\]^_`{|}~\n'
# initialising porter stemmer
stemmer = PorterStemmer()

# unique words will have all the unique words in the corpus
unique_words = []


# postinglist is the dictionary containg the keys as the word and value
# as another dictionary containing the keys as document ids and values as the frequency of the word
# present in the document.
postinglist = {}

# words_set is the list containg the lists of words for each document in the corpus
words_set = []

# from corpus english-corpora
file_folder = "english-corpora"
index = 0

# iterating through each document in the corpus
for file in glob.glob(file_folder + "\\*"):
    print(file)

    # opening and reading the doucment

    file = open(file, "r", encoding="utf8")
    data = file.read()
    # removing all the non ascii characters
    data = re.sub(r"[^\x00-\x7F]", " ", data)
    # lower case all the words in the document
    data = np.char.lower(data)

    # as symbols contains the punctuations..we remove all the punctuations from the document
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], " ")
        data = np.char.replace(data, "  ", " ")
    # replace ',' by' '
    data = np.char.replace(data, ",", " ")
    # removes apostrophies
    data = np.char.replace(data, "'", "")

    # removing stopwords
    data = remove_stop_words(data)
    # converting numbers to words
    data = convert_numbers(data)

    # stemming the document
    data = stemming(data)

    # as symbols contains the punctuations..we remove all the punctuations from the document

    # NOTE: WE AGAIN DO THIS BECAUSE CONVERITNG NUMBERS TO WORDS WILL GENERATE PUNCTUATIONS ALSO

    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], " ")
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ",", " ")

    # again we remove stopwords after numbers get converted to the words..it is needed

    data = remove_stop_words(data)
    # tokenizing the words
    words = word_tokenize(data)

    # append each word from the document to unique_words list.
    # also add it to the dictionary with the document id and the count.
    # if its already added then add the document id and its count to the postings of the word.
    for word in set(words):

        unique_words.append(word)

        if word not in postinglist:
            postinglist[word] = {}
            postinglist[word][doc_idx[index]] = words.count(word)

        else:
            postinglist[word][doc_idx[index]] = words.count(word)

    # append words in the words_set

    words_set.append(words)
    index = index + 1

# printing the time to preprocess.

print(time.time() - curr)


# In[23]:


# remove the dulpicate words from unique_words
unique_words = set(unique_words)


# In[24]:


# length of unique_words
len(unique_words)


# In[25]:


# length of postinglist
len(postinglist)


# as the length calculation in Question 3(b) will take time we will calculate that here only.

# In[21]:


# for Question 2 (b) , We need to find the length of the vector for each documents(using l2 norm)

# a single vector is a tfidf vector having the tfidf values for each of the word in that particular document.

# tfidf(of a word in a document)=tf* idf.

# where tf is the term frequency in that document and idf is the number of documents by the lenghth
# of the posting list for that particular word.

curr = time.time()
from math import *

# dictionary containing the length with respect to each document in key ,value form.
lenght_docs = {}
index = 0
# for each document
for doc_words in words_set:
    length = 0
    # length is calcualetd using tfidf of each word in that document
    for w in set(doc_words):
        # tf of the word
        tf = doc_words.count(w)
        # idf(normalised by log)
        idf = log2((len(doc_idx) + 1) / (len(postinglist[w]) + 1))
        # added to the length
        length += (tf * idf) ** 2
    # taking square root
    length = sqrt(length)
    # puting the document index and lenght in the form of key value pairs
    lenght_docs[index] = length
    print(index)
    index = index + 1
print(time.time() - curr)


# In[22]:


# storing the postinglist,words_set,unique_words, length_docs in a pickle file.

with open("Question1_output/posting_list", "wb") as handle:
    pickle.dump(postinglist, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("Question1_output/words_set", "wb") as handle:
    pickle.dump(words_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("Question1_output/unique_words", "wb") as handle:
    pickle.dump(unique_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("Question1_output/length_docs", "wb") as handle:
    pickle.dump(lenght_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


# In[ ]:


# In[ ]:
