import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def create_text_classification_model(vocab_size, max_len, num_classes):
    """
    Create a Keras Sequential model with 5 hidden dense layers 
    and softmax activation for text classification
    """
    model = Sequential([
        # Input layer: Embedding layer to convert words to vectors
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        
        # Flatten the 3D tensor to 2D
        Flatten(),
        
        # 5 hidden dense layers with dropout for regularization
        Dense(256, activation='relu'),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        Dropout(0.3),
        
        # Output layer with softmax activation
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example usage with dummy data
def prepare_data():
    # Example texts and labels (binary classification: positive/negative)
    texts = [
        "This movie was amazing and I loved it",
        "Terrible film, I hated everything about it",
        "The acting was good but the plot was confusing",
        "Best movie I've seen all year",
        "Waste of time and money"
    ]
    
    labels = [1, 0, 0, 1, 0]  # 1: positive, 0: negative
    
    # Tokenize texts
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences to ensure uniform length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    # Convert labels to categorical format for multi-class classification
    categorical_labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    
    return padded_sequences, categorical_labels, tokenizer.word_index, max_len

# Prepare example data
X, y, word_index, max_len = prepare_data()
vocab_size = len(word_index) + 1  # Add 1 for the reserved 0 index

# Create and display the model
model = create_text_classification_model(vocab_size, max_len, num_classes=2)
model.summary()

# Train the model (just for demonstration)
model.fit(X, y, epochs=5, batch_size=2, validation_split=0.2)