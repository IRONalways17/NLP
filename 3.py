import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def perform_stemming(text):
    # Initialize stemmers
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Apply stemming
    porter_stems = [porter.stem(word) for word in words]
    lancaster_stems = [lancaster.stem(word) for word in words]
    snowball_stems = [snowball.stem(word) for word in words]
    
    # Print results
    print("Original words:", words)
    print("\nPorter Stemmer:", porter_stems)
    print("\nLancaster Stemmer:", lancaster_stems)
    print("\nSnowball Stemmer:", snowball_stems)

# Example usage
sample_text = "running runner ran runs easily coding codes coded"
perform_stemming(sample_text)