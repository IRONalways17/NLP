import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import math

nltk.download('punkt')
nltk.download('stopwords')

def tf_idf(documents):
    """Implement TF-IDF algorithm for text representation"""
    # Preprocessing
    stop_words = set(stopwords.words('english'))
    processed_docs = []
    
    for doc in documents:
        # Tokenize
        tokens = word_tokenize(doc.lower())
        
        # Remove punctuation and stopwords
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        
        processed_docs.append(tokens)
    
    # Create vocabulary (unique words across all documents)
    all_words = [word for doc in processed_docs for word in doc]
    vocabulary = sorted(set(all_words))
    
    # Calculate Term Frequency (TF)
    tf_matrix = []
    for doc_tokens in processed_docs:
        doc_length = len(doc_tokens)
        tf_vector = []
        
        for term in vocabulary:
            term_count = doc_tokens.count(term)
            term_frequency = term_count / doc_length if doc_length > 0 else 0
            tf_vector.append(term_frequency)
        
        tf_matrix.append(tf_vector)
    
    # Calculate Inverse Document Frequency (IDF)
    idf_vector = []
    num_docs = len(documents)
    
    for term in vocabulary:
        doc_count = sum(1 for doc in processed_docs if term in doc)
        idf = math.log(num_docs / (1 + doc_count))  # Adding 1 to avoid division by zero
        idf_vector.append(idf)
    
    # Calculate TF-IDF Matrix
    tfidf_matrix = []
    for tf_vector in tf_matrix:
        tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vector)]
        tfidf_matrix.append(tfidf_vector)
    
    # Create DataFrame for better visualization
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=vocabulary)
    
    return tfidf_df, vocabulary

# Example usage
documents = [
    "The cat sits on the mat.",
    "The dog runs in the park.",
    "Cats and dogs are popular pets."
]

tfidf_matrix, vocab = tf_idf(documents)
print("Vocabulary:", vocab)
print("\nTF-IDF Matrix:")
print(tfidf_matrix)