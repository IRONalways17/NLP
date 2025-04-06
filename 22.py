import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import time
import os

# Display information for lab report
print(f"Current Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
try:
    print(f"Current User's Login: {os.getlogin()}")
except:
    print("Current User's Login: Unknown")
print("\n" + "="*50 + "\n")

def create_cbow_model(corpus, window_size=2, embedding_dim=100):
    """
    Create a Continuous Bag of Words (CBOW) model
    """
    # Tokenize the corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([" ".join(sentence) for sentence in corpus])
    vocab_size = len(tokenizer.word_index) + 1
    
    # Generate training data
    X = []  # Context words
    y = []  # Target words
    
    for sentence in corpus:
        # Convert words to indices
        word_indices = [tokenizer.word_index[word] for word in sentence if word in tokenizer.word_index]
        
        # Generate context and target pairs
        for i, target_word_idx in enumerate(word_indices):
            # Define context range
            start = max(0, i - window_size)
            end = min(len(word_indices), i + window_size + 1)
            context_indices = [word_indices[j] for j in range(start, end) if j != i]
            
            # Pad context to fixed size
            if len(context_indices) > 0:  # Only add if we have context words
                context_indices = pad_sequences([context_indices], maxlen=2*window_size, padding='pre')[0]
                X.append(context_indices)
                y.append(target_word_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode targets
    y_one_hot = keras.utils.to_categorical(y, num_classes=vocab_size)
    
    # Create the CBOW model
    model = Sequential([
        # Input is context words
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=X.shape[1]),
        
        # Average the context word embeddings
        GlobalAveragePooling1D(),
        
        # Output layer
        Dense(vocab_size, activation='softmax')
    ])
    
    # Display model architecture
    model.summary()
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # Train the model
    history = model.fit(X, y_one_hot, epochs=50, verbose=1)
    
    # Get word embeddings
    word_vectors = model.layers[0].get_weights()[0]
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'word_vectors': word_vectors,
        'history': history
    }

def visualize_embeddings(word_vectors, tokenizer, title, n_words=30):
    """
    Visualize word embeddings using t-SNE with dynamic perplexity
    """
    # Get words to visualize (all if less than n_words)
    words = list(tokenizer.word_index.keys())[:n_words]
    num_words = len(words)
    
    # Extract vectors for these words
    vectors = np.array([word_vectors[tokenizer.word_index[word]] for word in words])
    
    print(f"Visualizing {num_words} words with t-SNE...")
    
    # Set perplexity based on number of samples
    # t-SNE requires perplexity < n_samples
    perplexity = min(num_words - 1, 20)  # Use lower perplexity, maximum is n_words-1
    
    # Use t-SNE for dimensionality reduction with adjusted perplexity
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_vectors = tsne.fit_transform(vectors)
    
    # Plot words
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', alpha=0.5)
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), 
                    fontsize=9, alpha=0.7)
    
    plt.title(f'{title} Word Embeddings Visualization')
    plt.savefig(f"{title.lower().replace(' ', '_')}_visualization.png")
    plt.show()

def find_similar_words(word, word_vectors, tokenizer, top_n=5):
    """
    Find similar words based on cosine similarity
    """
    if word not in tokenizer.word_index:
        return []
    
    word_idx = tokenizer.word_index[word]
    word_vec = word_vectors[word_idx]
    
    # Calculate similarities
    similarities = {}
    for w, idx in tokenizer.word_index.items():
        if w != word:
            vec = word_vectors[idx]
            # Cosine similarity 
            similarity = np.dot(word_vec, vec) / (np.linalg.norm(word_vec) * np.linalg.norm(vec))
            similarities[w] = similarity
    
    similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return similar_words

# Example corpus
corpus = [
    "king queen royal palace crown throne kingdom",
    "man woman boy girl child person people",
    "apple orange banana fruit vegetable food eat",
    "car truck vehicle drive road transportation",
    "dog cat pet animal wild domestic",
    "computer laptop keyboard mouse technology device",
    "run walk jump move fast slow motion",
    "big small large tiny huge size",
    "hot cold warm cool temperature weather",
    "happy sad emotion feeling joy sorrow"
]

# Tokenize corpus
corpus = [sentence.lower().split() for sentence in corpus]

# Add more examples by shuffling and repeating
for _ in range(5):
    new_sentences = []
    for sentence in corpus:
        words = sentence.copy()
        random.shuffle(words)
        new_sentences.append(words)
    corpus.extend(new_sentences)

print("Creating CBOW Word2Vec model...")
cbow_results = create_cbow_model(corpus, window_size=2, embedding_dim=50)

# Get vocabulary size
vocab_size = len(cbow_results['tokenizer'].word_index)
print(f"Vocabulary size: {vocab_size} words")

# Visualize the embeddings
visualize_embeddings(cbow_results['word_vectors'], cbow_results['tokenizer'], 'CBOW')

# Find similar words for some test words
test_words = ['king', 'happy', 'computer', 'animal']
print("\nSimilar words:")
for word in test_words:
    similar = find_similar_words(word, cbow_results['word_vectors'], cbow_results['tokenizer'])
    print(f"\nWords similar to '{word}':")
    for w, sim in similar:
        print(f"  {w}: {sim:.4f}")

print("\nCBOW model implementation completed successfully!")