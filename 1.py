import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary resources
nltk.download('punkt')

def nltk_tokenization(text):
    # Sentence tokenization
    sentences = sent_tokenize(text)
    print("Sentence Tokenization:")
    for i, sentence in enumerate(sentences):
        print(f"{i+1}. {sentence}")
    
    # Word tokenization
    words = word_tokenize(text)
    print("\nWord Tokenization:")
    print(words)

# Example usage
sample_text = "Hello world! This is an example of tokenization. NLTK is a powerful NLP library."
nltk_tokenization(sample_text)