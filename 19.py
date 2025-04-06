# a) Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# b) Locate an open source dataset - using Kaggle IMDB Movie Reviews
print("Dataset: IMDB Movie Reviews for Sentiment Analysis")
print("Source: Kaggle - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Description: This dataset contains 50,000 movie reviews from IMDB, labeled as positive or negative.")

# c) Load the dataset
# For this example, we'll assume the dataset is downloaded as 'IMDB Dataset.csv'
try:
    df = pd.read_csv('IMDB Dataset.csv')
    print("\nDataset loaded successfully.")
except FileNotFoundError:
    print("\nDataset file not found. Using a smaller example dataset instead.")
    # Create a small example dataset if file not found
    reviews = [
        "This movie was fantastic! I really enjoyed it.",
        "Terrible film, complete waste of time.",
        "Great acting, but the plot was confusing.",
        "I fell asleep during this boring movie.",
        "One of the best films I've seen all year!"
    ]
    sentiments = ['positive', 'negative', 'positive', 'negative', 'positive']
    df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# d) Check for missing values and provide variable descriptions
print("\n--- Data Overview ---")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nVariable descriptions:")
for column in df.columns:
    if df[column].dtype == 'object':
        print(f"- {column}: Text data with {df[column].nunique()} unique values")
    else:
        print(f"- {column}: Numeric data with range {df[column].min()} to {df[column].max()}")

# e) Data formatting and normalization
print("\n--- Data Formatting and Normalization ---")

# Function to clean text data
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning function to reviews
df['cleaned_review'] = df['review'].apply(clean_text)

print("Added cleaned version of text data:")
print(df[['review', 'cleaned_review']].head(2))

# f) Turn categorical variables into quantitative variables
print("\n--- Categorical to Quantitative Conversion ---")

# Label encoding for the sentiment column
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])

print("Encoded sentiment values:")
for i, label in enumerate(label_encoder.classes_):
    print(f"- {label}: {i}")

# Create basic features from text
df['review_length'] = df['cleaned_review'].apply(len)
df['word_count'] = df['cleaned_review'].apply(lambda x: len(x.split()))

# Add sentiment polarity score (simplified example)
df['positive_words'] = df['cleaned_review'].apply(
    lambda x: sum(1 for word in x.split() if word in ['good', 'great', 'excellent', 'best', 'liked', 'enjoyed'])
)
df['negative_words'] = df['cleaned_review'].apply(
    lambda x: sum(1 for word in x.split() if word in ['bad', 'worst', 'terrible', 'awful', 'boring', 'waste'])
)
df['sentiment_score'] = df['positive_words'] - df['negative_words']

print("\nFinal processed dataframe:")
print(df.head())

# Summary statistics of numerical features
print("\nSummary statistics of numerical features:")
print(df[['review_length', 'word_count', 'positive_words', 'negative_words', 'sentiment_score']].describe())

# Display correlation between numerical features
print("\nCorrelation between numerical features and sentiment:")
print(df[['review_length', 'word_count', 'positive_words', 'negative_words', 'sentiment_score', 'sentiment_encoded']].corr()['sentiment_encoded'])