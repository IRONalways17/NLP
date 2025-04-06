import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
import random

def create_skipgram_model(corpus, window_size=2, embedding_dim=100, negative_samples=5):
    """
    Create a Skip-gram model using Keras to predict context words given a target word
    """
    # Tokenize the corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([corpus])
    
    # Get vocabulary size and word index
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
    word_index = tokenizer.word_index
    
    # Convert corpus to sequences
    sequences = tokenizer.texts_to_sequences([corpus])[0]
    
    # Prepare training data: target word (X) and context word (y) pairs
    word_pairs = []
    
    for i in range(len(sequences)):
        target_word = sequences[i]
        
        # Get context words within the window
        for j in range(max(0, i - window_size), min(len(sequences), i + window_size + 1)):
            if i != j:  # Skip the target word itself
                context_word = sequences[j]
                word_pairs.append((target_word, context_word))
                
                # Add negative samples
                for _ in range(negative_samples):
                    # Select a random word from vocabulary as negative sample
                    negative_word = random.randint(1, vocab_size - 1)
                    while negative_word == target_word or negative_word == context_word:
                        negative_word = random.randint(1, vocab_size - 1)
                    
                    # Add negative sample with label 0
                    word_pairs.append((target_word, negative_word))
    
    # Shuffle the word pairs
    random.shuffle(word_pairs)
    
    # Separate target words and context words
    target_words, context_words = zip(*word_pairs)
    
    # Convert to numpy arrays
    X = np.array(target_words)
    y = np.array(context_words)
    
    # Define the Skip-gram model
    model = Sequential([
        # Input is the target word
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),
        
        # Flatten the embedding
        Flatten(),
        
        # Hidden layer
        Dense(128, activation='relu'),
        
        # Output layer to predict context word
        Dense(vocab_size, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(model.summary())
    
    # Train the model
    model.fit(X.reshape(-1, 1), y, epochs=50, batch_size=128, verbose=1)
    
    # Extract word embeddings
    word_embeddings = model.get_weights()[0]
    
    # Create a dictionary mapping words to their embeddings
    word_embedding_dict = {}
    for word, i in word_index.items():
        word_embedding_dict[word] = word_embeddings[i]
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'word_embeddings': word_embedding_dict,
        'vocab_size': vocab_size
    }

# Example usage
corpus = """Natural language processing is a subfield of linguistics computer science 
and artificial intelligence concerned with the interactions between computers and human language. 
The goal is to enable computers to process and understand natural language."""

# Create the Skip-gram model
skipgram_results = create_skipgram_model(corpus, window_size=2, embedding_dim=50)

# Test the model with example target word
tokenizer = skipgram_results['tokenizer']
model = skipgram_results['model']

# Example: Predict context words for a given target word
def predict_context(target_word, model, tokenizer, top_n=5):
    # Convert target word to sequence
    target_sequence = tokenizer.texts_to_sequences([target_word])
    
    if not target_sequence[0]:
        return "Target word not in vocabulary"
    
    # Predict
    prediction = model.predict(np.array([target_sequence[0]]))
    
    # Get top N predicted context words
    top_indices = prediction[0].argsort()[-top_n:][::-1]
    
    # Convert indices back to words
    reverse_word_map = {v: k for k, v in tokenizer.word_index.items()}
    top_words = [reverse_word_map.get(i) for i in top_indices]
    
    return top_words

# Example target word
target = "language"
predicted_context = predict_context(target, model, tokenizer)
print(f"Target word: {target}")
print(f"Predicted context words: {predicted_context}")