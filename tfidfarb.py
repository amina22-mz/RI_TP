# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:08:43 2023

@author: amina
"""

import re
from farasa.segmenter import FarasaSegmenter
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer  # Import ISRIStemmer
from nltk.tokenize import word_tokenize
#%%
def preprocess_arabic_text(text):
    # Initialize the Farasa segmenter
    segmenter = FarasaSegmenter()

    # Initialize the ISRI Arabic stemmer
    stemmer = ISRIStemmer()

    # Load Arabic stopwords
    stop_words = set(stopwords.words('arabic'))

    # Tokenize the text using Farasa
    tokens = segmenter.segment(text).split()

    processed_words = []

    for token in tokens:
        # Remove any digits
        token = re.sub(r'\d', '', token)

        # Remove diacritics
        token = re.sub(r'[\u064b-\u065f\u0640]', '', token)

        # Convert to lowercase
        token = token.lower()

        # Remove punctuation and non-Arabic characters
        token = re.sub(r'[^ء-ي\s]', '', token)

        # Check if the token is a stop word or too short
        if token not in stop_words and len(token) > 2:
            # Apply stemming using the ISRI Arabic stemmer
            stemmed_token = stemmer.stem(token)
            processed_words.append(stemmed_token)

    return processed_words

#%%
def readdoc(link):
    try:
        with open(link, 'r', encoding='utf-8') as f:
            # Read all lines from the document
            lines = f.readlines()
            doc = []

            for line in lines:
                # Remove newline and other unwanted characters
                line = line.strip().lower()

                # Perform preprocessing using the preprocess_arabic_text function
                processed_words = preprocess_arabic_text(line)

                # Append the list of processed words to the document
                doc.extend(processed_words)

            return doc

    except Exception as e:
        print(f"Error reading document from {link}: {str(e)}")
        return []
#%%
def read_corpus(n, folderLink):
    documents = []
    for i in range(1,n+1):  # Start the loop from 1, not 0
        # Read the document from the specified folderLink
        document = readdoc('%s/%s.txt' % (folderLink, i))

        # Append the preprocessed document to the list of documents
        documents.append(document)

    return documents
corpus=read_corpus(2,'C:/Users/amina/RI_TP/arabic')


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