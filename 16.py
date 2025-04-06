import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

def bag_of_words(documents):
    """Implement Bag of Words algorithm for text representation"""
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
    
    # Create document-term matrix
    bow_matrix = []
    
    for doc_tokens in processed_docs:
        # Count word frequencies
        word_counts = Counter(doc_tokens)
        
        # Create vector based on vocabulary
        doc_vector = [word_counts.get(word, 0) for word in vocabulary]
        bow_matrix.append(doc_vector)
    
    # Create DataFrame for better visualization
    bow_df = pd.DataFrame(bow_matrix, columns=vocabulary)
    
    return bow_df, vocabulary

# Example usage
documents = [
    "The cat sits on the mat.",
    "The dog runs in the park.",
    "Cats and dogs are popular pets."
]

bow_matrix, vocab = bag_of_words(documents)
print("Vocabulary:", vocab)
print("\nBag of Words Matrix:")
print(bow_matrix)