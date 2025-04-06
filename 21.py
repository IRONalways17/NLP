import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text data for classification"""
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

# Create a sample dataset for demonstration
def create_sample_dataset():
    texts = [
        "I love this phone. It has great features and battery life.",
        "This movie was fantastic and entertaining.",
        "The food at this restaurant is delicious and affordable.",
        "I hate this laptop. It's slow and crashes frequently.",
        "This book was boring and poorly written.",
        "The service at this hotel was terrible and unprofessional.",
        "This phone has an excellent camera and fast processor.",
        "The movie had great actors but a predictable plot.",
        "The restaurant offered a variety of menu options.",
        "This laptop has a short battery life and gets hot quickly.",
        "The book had interesting characters but a confusing storyline.",
        "The hotel room was dirty and uncomfortable."
    ]
    
    labels = [
        'positive', 'positive', 'positive',
        'negative', 'negative', 'negative',
        'positive', 'neutral', 'neutral',
        'negative', 'neutral', 'negative'
    ]
    
    return pd.DataFrame({'text': texts, 'sentiment': labels})

# Main function for text classification
def naive_bayes_text_classification():
    # Get dataset
    df = create_sample_dataset()
    print(f"Dataset created with {len(df)} examples")
    
    # Preprocess the text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['sentiment'], test_size=0.3, random_state=42
    )
    
    # Create document-term matrix using bag-of-words model
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_counts, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_counts)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Test with new examples
    new_texts = [
        "This product exceeded my expectations",
        "I regret buying this item, complete waste of money",
        "It was okay, nothing special"
    ]
    
    # Preprocess and transform the new texts
    new_processed = [preprocess_text(text) for text in new_texts]
    new_counts = vectorizer.transform(new_processed)
    
    # Predict sentiment
    new_predictions = classifier.predict(new_counts)
    
    print("\nPredictions for new texts:")
    for text, sentiment in zip(new_texts, new_predictions):
        print(f"Text: '{text}'")
        print(f"Predicted sentiment: {sentiment}\n")

# Run the function
naive_bayes_text_classification()