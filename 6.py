import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download necessary data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def perform_pos_tagging(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    try:
        # Try standard POS tagging
        tagged_words = pos_tag(tokens)
    except LookupError:
        try:
            # Fallback to universal tagset
            tagged_words = pos_tag(tokens, tagset='universal')
            print("Note: Using universal tagset instead of standard Penn Treebank tagset")
        except:
            # Simplest fallback - just mark everything as noun (NN)
            tagged_words = [(word, "NN") for word in tokens]
            print("Warning: Using simplified tagging (all words marked as nouns)")
    
    # Print results
    print("Original text:")
    print(text)
    print("\nPOS Tagged words:")
    for word, tag in tagged_words:
        print(f"{word}: {tag}")
    
    # Simplified explanation of some common tags
    tag_descriptions = {
        'NN': 'Noun, singular',
        'NNS': 'Noun, plural',
        'VB': 'Verb, base form',
        'VBD': 'Verb, past tense',
        'JJ': 'Adjective',
        'RB': 'Adverb',
        'DT': 'Determiner',
        'IN': 'Preposition'
    }
    
    print("\nCommon POS tag meanings:")
    for tag, desc in tag_descriptions.items():
        print(f"{tag}: {desc}")

# Example usage
sample_text = "The quick brown fox jumps over the lazy dog"
perform_pos_tagging(sample_text)