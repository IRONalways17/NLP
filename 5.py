import spacy
from spacy.lang.en.stop_words import STOP_WORDS

def find_stopwords(text):
    # Load English language model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Find stopwords
    stopwords_found = [token.text for token in doc if token.text.lower() in STOP_WORDS]
    
    # Print results
    print("Original text:")
    print(text)
    print("\nStopwords found:")
    print(stopwords_found)
    print(f"\nTotal stopwords: {len(stopwords_found)}")
    
    # Print all stopwords in spaCy (limited to first 20)
    print("\nSample of stopwords in spaCy:")
    print(list(STOP_WORDS)[:20], "...")

# Example usage
sample_text = "This is an example of a text that contains many common stopwords and we will identify them."
find_stopwords(sample_text)