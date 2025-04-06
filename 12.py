import nltk
from nltk import word_tokenize, pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

# Download necessary data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_named_entities(text):
    # Tokenize and tag the words
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    
    # Apply NER
    ner_tree = ne_chunk(tagged_tokens)
    
    # Extract named entities
    named_entities = []
    for subtree in ner_tree:
        if isinstance(subtree, Tree):
            entity_type = subtree.label()
            entity_text = ' '.join([word for word, tag in subtree.leaves()])
            named_entities.append((entity_text, entity_type))
    
    return named_entities

# Example usage
text = "Apple Inc. was founded by Steve Jobs in California. Microsoft is another company based in the United States."
entities = extract_named_entities(text)

print("Named Entities:")
for entity, entity_type in entities:
    print(f"{entity} - {entity_type}")