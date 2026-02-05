# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:57:48 2023

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
    try:
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

    except Exception as e:
        print(f"Error ")
        return []

#%%
def read_corpus(n, folderLink):
    documents = []
    for i in range(1, n + 1):  # Start the loop from 1, not 0
        # Read the document from the specified folderLink
        document = readdoc('%s/%s.txt' % (folderLink, i))

        # Append the preprocessed document to the list of documents
        documents.append(document)

    return documents
K=read_corpus(2,'C:/Users/amina/RI_TP/english')
print(K)
#%%
#create inverted index and calculate stem frequencies


def create_alphabetical_index(corpus):
    index = {} # create dictionary 
    sorted_terms = sorted(set(term for document in corpus for term in document)) #stored list contain the tokens of the corpus
     # set() store multiple item in a string
     # stored make the items in order
    for term in sorted_terms:
        term_doc_ids = [] #initialize a liste fore the document IDs where the current term appears
        for doc_id, document in enumerate(corpus):
            if term in document:
                term_doc_ids.append(doc_id)
        index[term] = term_doc_ids #after check all document add term_doc_ids to the index , and the term is the key
    
    return index



# Create an alphabetical inverted index for English documents
english_corpus = read_corpus(20, 'C:/Users/amina/RI_TP/english')
english_alphabetical_index = create_alphabetical_index(english_corpus)

stem_frequency_dict = defaultdict(dict) #initialiser un dictionaire to dtore the stem frequency for each term in each doc 

for term, doc_ids in english_alphabetical_index.items():
    for doc_id in doc_ids:
        document_name = f'Document {doc_id + 1}'  # Assuming document names are "Document 1", "Document 2", etc.
        stem_frequency_dict[document_name][term] = english_corpus[doc_id].count(term)

# Convert the stem frequency dictionary to a DataFrame
index_df = pd.DataFrame(stem_frequency_dict).transpose()
index_df.index = range(1, 21)  # Rename the index to start from 1 to 20

# Sort the DataFrame by terms in alphabetical order
index_df.sort_index(axis=1, inplace=True)

# Define the output CSV file path
output_csv_file = 'english_inverted_index.csv'

# Save the DataFrame as a CSV file
index_df.to_csv(output_csv_file)

print(f"Alphabetical inverted index saved to {output_csv_file}")

