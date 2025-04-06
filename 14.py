import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

def extract_ngrams(corpus, n_values=[1, 2, 3]):
    # Tokenize the corpus
    tokens = word_tokenize(corpus.lower())
    
    # Extract ngrams for each value of n
    results = {}
    
    for n in n_values:
        # Generate n-grams
        n_grams = list(ngrams(tokens, n))
        
        # Count frequencies
        ngram_freq = Counter(n_grams)
        
        # Format the n-grams for readability
        formatted_ngrams = {}
        for gram, count in ngram_freq.most_common():
            gram_text = ' '.join(gram)
            formatted_ngrams[gram_text] = count
        
        # Store the results
        if n == 1:
            results['Unigrams'] = formatted_ngrams
        elif n == 2:
            results['Bigrams'] = formatted_ngrams
        elif n == 3:
            results['Trigrams'] = formatted_ngrams
        else:
            results[f'{n}-grams'] = formatted_ngrams
    
    return results

# Example usage
corpus = """Natural language processing is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and human language. 
The goal is to enable computers to process and understand natural language."""

ngram_results = extract_ngrams(corpus)

# Print results
for ngram_type, ngram_dict in ngram_results.items():
    print(f"\n{ngram_type}:")
    for i, (gram, count) in enumerate(ngram_dict.items()):
        print(f"  {gram}: {count}")
        if i >= 9:  # Show only top 10
            print(f"  ... and {len(ngram_dict) - 10} more")
            break