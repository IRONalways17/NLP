import spacy

def spacy_tokenization(text):
    # Load English language model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Sentence segmentation
    print("Sentence Segmentation:")
    for i, sent in enumerate(doc.sents):
        print(f"{i+1}. {sent}")
    
    # Word tokenization
    print("\nWord Tokenization:")
    tokens = [token.text for token in doc]
    print(tokens)

# Example usage
sample_text = "Hello world! This is an example of tokenization using spaCy. It handles sentence boundaries well."
spacy_tokenization(sample_text)