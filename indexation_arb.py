# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 11:27:59 2023

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


# Example usage
arabic_text = "قام الطالب بالدراسة في المكتبة"
processed_arabic_words = preprocess_arabic_text(arabic_text)
print(processed_arabic_words)
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
K=read_corpus(2,'C:/Users/amina/RI_TP/arabic')
print(K)
#%%
import pandas as pd
from collections import defaultdict

def create_alphabetical_index(corpus):
    index = {}
    sorted_terms = sorted(set(term for document in corpus for term in document))
    
    for term in sorted_terms:
        term_doc_ids = []
        for doc_id, document in enumerate(corpus):
            if term in document:
                term_doc_ids.append(doc_id)
        index[term] = term_doc_ids
    
    return index
#%%

# Create an alphabetical inverted index for English documents
arabic_corpus = read_corpus(10, 'C:/Users/amina/RI_TP/arabic')
arabic_alphabetical_index = create_alphabetical_index(arabic_corpus)

# Create a dictionary to store the stem frequency for each document
stem_frequency_dict = defaultdict(dict)

for term, doc_ids in arabic_alphabetical_index.items():
    for doc_id in doc_ids:
        document_name = f'Document {doc_id + 1}'  # Assuming document names are "Document 1", "Document 2", etc.
        stem_frequency_dict[document_name][term] = arabic_corpus[doc_id].count(term)

# Convert the stem frequency dictionary to a DataFrame
index_df = pd.DataFrame(stem_frequency_dict).transpose()
index_df.index = range(1, 11)  # Rename the index to start from 1 to 20

# Sort the DataFrame by terms in alphabetical order
index_df.sort_index(axis=1, inplace=True)

# Define the output CSV file path
output_csv_file = 'arabic_inverted_index.csv'

# Save the DataFrame as a CSV file
index_df.to_csv(output_csv_file, encoding='utf-8')

print(f"Alphabetical inverted index saved to {output_csv_file}")





