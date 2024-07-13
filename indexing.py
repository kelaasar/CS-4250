#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: tfidf_document_term_matrix.py
# SPECIFICATION: This program reads a file collection.csv and outputs the tf-idf document-term matrix.
# FOR: CS 4250- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

# Importing some Python libraries
import csv
import math

# Reading the data in a csv file
documents = []

with open('collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            documents.append(row[0])

# Conducting stopword removal for pronouns/conjunctions. Using a set to define stopwords.
stopWords = {"i", "she", "he", "it", "we", "they", "you", "and", "or", "but"}

# Conducting stemming. Using a dictionary to map word variations to their stem.
stemming = {
    "cats": "cat",
    "dogs": "dog",
    "loves": "love",
    "loving": "love",
    "loved": "love"
}

# Function to preprocess a document (remove stopwords and apply stemming)
def preprocess_document(doc):
    words = doc.lower().split()
    processed_words = []
    for word in words:
        if word not in stopWords:
            stemmed_word = stemming.get(word, word)
            processed_words.append(stemmed_word)
    return processed_words

# Preprocessing all documents
preprocessed_documents = [preprocess_document(doc) for doc in documents]

# Identifying the index terms (unique terms)
terms = []
for doc in preprocessed_documents:
    for word in doc:
        if word not in terms:
            terms.append(word)

# Building the document-term matrix by using the tf-idf weights
docTermMatrix = []

# Calculating TF
def term_frequency(term, document):
    return document.count(term) / len(document)

# Calculating IDF
def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log((1 + len(all_documents)) / (1 + num_docs_containing_term)) + 1

# Calculating TF-IDF
for doc in preprocessed_documents:
    tf_idf_vector = []
    for term in terms:
        tf = term_frequency(term, doc)
        idf = inverse_document_frequency(term, preprocessed_documents)
        tf_idf_value = tf * idf
        tf_idf_vector.append(round(tf_idf_value, 3))
    docTermMatrix.append(tf_idf_vector)

# Printing the document-term matrix
print("Document-Term Matrix:")
for row in docTermMatrix:
    print(row)
