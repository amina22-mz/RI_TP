# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:10:34 2023

@author: amina
"""

import re #library used to regular expression 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import os
import pandas as pd
from collections import defaultdict # dictionary-like data structure that allows you to provide a default value for keys that don't exist in the dictionary.

nltk.download('punkt')
nltk.download('stopwords')
#%%
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    processed_words = []

    for word in words:
        if not word.isdigit(): #if the word is not number
            word = word.lower()
            stemmed_word = stemmer.stem(word) #the root of the word
            letters_only = re.findall(r'[a-zA-Z]+', stemmed_word) #delete chifre ,extract only the alphabetic characters
            
            if letters_only: #the list letter only is not empty
                cleaned_word = ' '.join(letters_only) #join the words from the letters only list in a string, between words thers is space 
                
                if cleaned_word not in stop_words and len(cleaned_word) > 2:
                    processed_words.append(cleaned_word) #add the word to the proccess words list

    return processed_words
#%%
def readdoc(link):
  
        with open(link, 'r', encoding='utf-8') as f: #open the specefic document
            # Read all lines from the document
            lines = f.readlines()
            doc = []

            for line in lines:
                
                line = line.strip().lower() #remove leading and trailing whitespace from each line and converts the line to lowercase.

                # Tokenize the cleaned line into words using word_tokenize
                words = word_tokenize(line)

                # Perform preprocessing using the preprocess_text function
                processed_words = preprocess_text(line)

                # Append the list of processed words to the document
                doc.extend(processed_words)

            return doc
d=readdoc('C:/Users/amina/RI_TP/english/1.txt')
print(d)

#%%
def read_corpus(n, folderLink):
    documents = []
    for i in range(1, n + 1):
        # Read the document from the specified folderLink
        document = readdoc('%s/%s.txt' % (folderLink, i))

        # Append the preprocessed document to the list of documents
        documents.append(document)

    return documents

# Usage: Read the corpus
corpus = read_corpus(2, 'C:/Users/amina/RI_TP/english')
print(corpus)
#%%
def bag_of_words(documents):
    bw = {}
    
    for lines in documents:
        for word in lines:
            if word in bw:
                bw[word] += 1
            else:
                bw[word] = 1

    return bw

# Usage: Create a Bag of Words representation for the entire corpus
bw = bag_of_words(corpus)
print(bw)

#%%
def frequency_As_Matrix():
    documents  = read_corpus(2,'C:/Users/amina/RI_TP/english')
    for doc in  documents :
        bag_Of_w = bag_of_words(documents)
        for w in bag_Of_w:
            print('%s = %d,' %(w,bag_Of_w.get(w)),end ='\t')
        print('\n\n')
print(frequency_As_Matrix())
matrix = frequency_As_Matrix()

#%%
def word_In_Doc(mw,doc):
    for line in doc:
        for w in line:
                    if w.lower()== mw.lower():
                        return True
    return False
print(word_In_Doc('studi',corpus))

#%%
from math import log
def tfidf(corpus):
    wtf = {}
    documents = corpus
    for doc in documents:
        tfAll =  bag_of_words(documents)
        for w in tfAll:
            df=1
            for d in documents:
                if word_In_Doc(w,d):
                    df+=1   
            wtf[w]=(1+log(tfAll[w]))*log(20/df)
    return wtf

english_corpus = read_corpus(2, 'C:/Users/amina/RI_TP/english')
print(tfidf(english_corpus))

#%%

def create_alphabetical_index(corpus, tfidf_scores):
    index = {}  # Create an empty dictionary to store the index

    for doc_id, document in enumerate(corpus):
        for term in document:
            if term not in index:
                index[term] = {}
            index[term][f'Document {doc_id + 1}'] = tfidf_scores.get((term, doc_id), 0)

    return index

# Read the English corpus
english_corpus = read_corpus(20, 'C:/Users/amina/RI_TP/english')

# Calculate TF-IDF scores for the English corpus
tfidf_scores = tfidf(english_corpus)

# Create an alphabetical inverted index for the English documents
alphabetical_index = create_alphabetical_index(english_corpus, tfidf_scores)

# Create a DataFrame from the alphabetical index
index_df = pd.DataFrame.from_dict(alphabetical_index, orient='index').fillna(0).astype(float)

# Sort the DataFrame by terms in alphabetical order
index_df = index_df.sort_index(axis=0)

# Save the DataFrame as a CSV file
index_df.to_csv('english_index.csv')

print("CSV file saved.")