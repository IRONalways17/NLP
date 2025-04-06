import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import math

nltk.download('punkt')

def calculate_probability(statement, corpus, n=2):
    """Calculate probability of a statement using n-gram language model"""
    # Tokenize corpus and statement
    corpus_tokens = word_tokenize(corpus.lower())
    statement_tokens = word_tokenize(statement.lower())
    
    # Generate n-grams and (n-1)-grams for corpus
    corpus_ngrams = list(ngrams(corpus_tokens, n))
    corpus_n_minus_1_grams = list(ngrams(corpus_tokens, n-1))
    
    # Count frequencies
    ngram_counts = Counter(corpus_ngrams)
    n_minus_1_gram_counts = Counter(corpus_n_minus_1_grams)
    
    # Calculate statement probability
    log_probability = 0
    
    for i in range(len(statement_tokens) - n + 1):
        current_ngram = tuple(statement_tokens[i:i+n])
        current_n_minus_1_gram = tuple(statement_tokens[i:i+n-1])
        
        # Calculate conditional probability P(wn|w1...wn-1)
        ngram_count = ngram_counts[current_ngram]
        n_minus_1_gram_count = n_minus_1_gram_counts[current_n_minus_1_gram]
        
        # Apply smoothing to avoid zero probabilities
        smoothed_prob = (ngram_count + 1) / (n_minus_1_gram_count + len(set(corpus_tokens)))
        
        # Use log probabilities to avoid underflow
        log_probability += math.log2(smoothed_prob) if smoothed_prob > 0 else -float('inf')
    
    return 2 ** log_probability  # Convert back from log space

# Example usage
corpus = """This is my house. This is my car. This is my dog. 
This cat is cute. The cat is black. I love my cat. This is her cat."""

statement = "This is my cat"

probability = calculate_probability(statement, corpus, n=2)
print(f"Probability of '{statement}' using bigram model: {probability:.10f}")