import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

def remove_stopwords(text):
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Tokenize text
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    print("Original text:")
    print(text)
    print("\nStopwords removed:")
    print(' '.join(filtered_words))
    print(f"\nRemoved {len(words) - len(filtered_words)} stopwords")

# Example usage
sample_text = "This is an example sentence showing how to remove the stopwords from a text."
remove_stopwords(sample_text)