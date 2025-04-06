import spacy
import os
import time

def extract_named_entities(text):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Process text
    doc = nlp(text)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Example usage
if __name__ == "__main__":
    # Display information similar to your output
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        print(f"Current User's Login: {os.getlogin()}")
    except:
        print("Current User's Login: Unknown")
    print("\n" + "="*50 + "\n")
    
    # Sample text
    text = "Apple Inc. was founded by Steve Jobs in California in 1976. Microsoft was founded by Bill Gates and Paul Allen in 1975."
    
    # Extract entities
    entities = extract_named_entities(text)
    
    # Display results
    print("Named Entities:")
    for entity, label in entities:
        print(f"  - {entity}: {label}")
    
    print("\nEntity Types:")
    print("  - PERSON: People")
    print("  - ORG: Organizations")
    print("  - GPE: Countries, cities")
    print("  - DATE: Dates")
    print("  - CARDINAL: Numbers")