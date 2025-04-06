import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary data
nltk.download('punkt')
nltk.download('wordnet')

def perform_lemmatization(text):
    # NLTK lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    
    nltk_lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    nltk_lemmas_v = [lemmatizer.lemmatize(word, pos='v') for word in tokens]  # verb lemmatization
    
    # spaCy lemmatization
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    spacy_lemmas = [token.lemma_ for token in doc]
    
    # Print results
    print("Original text:", text)
    print("\nNLTK Lemmatization (default):", nltk_lemmas)
    print("NLTK Lemmatization (verbs):", nltk_lemmas_v)
    print("spaCy Lemmatization:", spacy_lemmas)

# Example usage
sample_text = "running studies better wolves women are going children"
perform_lemmatization(sample_text)