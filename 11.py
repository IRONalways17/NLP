import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser

# Download necessary data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def perform_chunking(text):
    # Tokenize and tag the words
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)
    
    # Define a grammar for chunking
    grammar = r"""
      NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
      PP: {<IN><NP>}               # Chunk prepositions followed by NP
      VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
      CLAUSE: {<NP><VP>}           # Chunk NP, VP
    """
    
    # Create a chunk parser
    chunk_parser = RegexpParser(grammar)
    
    # Parse the tagged words
    tree = chunk_parser.parse(tagged_words)
    
    # Print results
    print("Chunking results:")
    for subtree in tree.subtrees():
        if subtree.label() != 'S':  # Skip the root
            print(f"{subtree.label()}: {' '.join([word for word, tag in subtree.leaves()])}")
    
    # Draw the tree (uncomment if needed)
    # tree.draw()
    
    return tree

# Example usage
text = "The quick brown fox jumps over the lazy dog. Natural language processing is fascinating."
chunked_tree = perform_chunking(text)