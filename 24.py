import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import time

# Display current time and user for lab report purposes
print(f"Current Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
try:
    import os
    print(f"Current User's Login: {os.getlogin()}")
except:
    print("Current User's Login: Unknown")
print("\n" + "="*50 + "\n")

def create_word2vec_model(corpus, model_type='cbow', embedding_dim=100, window_size=2):
    """
    Create a Word2Vec model using either CBOW or Skip-gram architecture
    
    Args:
        corpus: List of tokenized sentences
        model_type: 'cbow' or 'skipgram'
        embedding_dim: Dimension of word embeddings
        window_size: Context window size
    
    Returns:
        Dictionary with model, tokenizer, and word vectors
    """
    # Flatten the corpus and create vocabulary
    all_words = [word for sentence in corpus for word in sentence]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    
    print(f"Building {model_type.upper()} Word2Vec model with vocabulary size: {vocab_size}")
    
    # Generate training data
    X = []
    y = []
    
    for sentence in corpus:
        # Convert words to indices
        word_indices = [tokenizer.word_index[word] for word in sentence if word in tokenizer.word_index]
        
        # Generate context and target pairs
        for i, target_word_idx in enumerate(word_indices):
            # Define context range
            start = max(0, i - window_size)
            end = min(len(word_indices), i + window_size + 1)
            context_indices = [word_indices[j] for j in range(start, end) if j != i]
            
            # Pad context to fixed size (2 * window_size)
            context_indices = pad_sequences([context_indices], maxlen=2*window_size, padding='pre')[0]
            
            if model_type == 'cbow':
                X.append(context_indices)
                y.append(target_word_idx)
            else:  # Skip-gram
                for context_idx in context_indices:
                    if context_idx > 0:  # Skip padding
                        X.append(target_word_idx)
                        y.append(context_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode targets
    y_one_hot = keras.utils.to_categorical(y, num_classes=vocab_size)
    
    # Create model
    if model_type == 'cbow':
        # CBOW model: predict target word from context words
        input_layer = Input(shape=(X.shape[1],))
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
        
        # Use GlobalAveragePooling1D instead of Lambda with K.mean
        pooling_layer = GlobalAveragePooling1D()(embedding_layer)
        
        output_layer = Dense(vocab_size, activation='softmax')(pooling_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
    else:
        # Skip-gram model: predict context words from target word
        input_layer = Input(shape=(1,))
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
        reshape_layer = keras.layers.Reshape((embedding_dim,))(embedding_layer)
        output_layer = Dense(vocab_size, activation='softmax')(reshape_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # Print model summary
    print(f"\n{model_type.upper()} Word2Vec Model Architecture:")
    model.summary()
    
    # Train model
    print(f"\nTraining {model_type.upper()} Word2Vec model...")
    history = model.fit(X, y_one_hot, epochs=50, verbose=1)
    
    # Extract word vectors from embedding layer
    word_vectors = model.get_layer('embedding').get_weights()[0]
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'word_vectors': word_vectors,
        'history': history
    }

def visualize_embeddings(word_vectors, tokenizer, title, n_words=50):
    """
    Visualize word embeddings using t-SNE
    """
    # Get most common words
    words = list(tokenizer.word_index.keys())[:n_words]
    vectors = np.array([word_vectors[tokenizer.word_index[word]] for word in words])
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    
    # Plot words in 2D space
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', alpha=0.5)
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), 
                    fontsize=9, alpha=0.7)
    
    plt.title(f'{title} - Word Embeddings Visualization (t-SNE)')
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}_embeddings.png")
    plt.show()

def find_similar_words(word, word_vectors, tokenizer, top_n=5):
    """
    Find similar words based on cosine similarity
    """
    if word not in tokenizer.word_index:
        return []
    
    word_idx = tokenizer.word_index[word]
    word_vec = word_vectors[word_idx]
    
    # Calculate cosine similarity with all words
    similarities = {}
    for w, idx in tokenizer.word_index.items():
        if w != word:
            vec = word_vectors[idx]
            # Cosine similarity
            similarity = np.dot(word_vec, vec) / (np.linalg.norm(word_vec) * np.linalg.norm(vec))
            similarities[w] = similarity
    
    # Sort by similarity and get top N
    similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return similar_words

# Generate example corpus
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

# Add more sentences to make the corpus larger
for _ in range(10):
    new_sentences = []
    for sentence in corpus:
        words = sentence.copy()
        random.shuffle(words)
        new_sentences.append(words)
    corpus.extend(new_sentences)

# Create and evaluate models
print("Creating Word2Vec models...")

# CBOW model
cbow_results = create_word2vec_model(corpus, model_type='cbow', embedding_dim=50)

# Skip-gram model
skipgram_results = create_word2vec_model(corpus, model_type='skipgram', embedding_dim=50)

# Visualize embeddings
visualize_embeddings(cbow_results['word_vectors'], cbow_results['tokenizer'], 
                     'CBOW Word2Vec', n_words=30)
visualize_embeddings(skipgram_results['word_vectors'], skipgram_results['tokenizer'], 
                     'Skip-gram Word2Vec', n_words=30)

# Find similar words examples
test_words = ['king', 'happy', 'computer', 'animal']
print("\nSimilar words based on CBOW model:")
for word in test_words:
    similar = find_similar_words(word, cbow_results['word_vectors'], cbow_results['tokenizer'])
    print(f"\nWords similar to '{word}':")
    for w, sim in similar:
        print(f"  {w}: {sim:.4f}")

print("\nSimilar words based on Skip-gram model:")
for word in test_words:
    similar = find_similar_words(word, skipgram_results['word_vectors'], skipgram_results['tokenizer'])
    print(f"\nWords similar to '{word}':")
    for w, sim in similar:
        print(f"  {w}: {sim:.4f}")

print("\nWord2Vec implementation completed successfully!")