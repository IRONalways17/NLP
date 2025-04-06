import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

def find_document_similarity(doc1, doc2):
    """
    Calculate similarity between two documents using different measures:
    1. Cosine similarity of TF-IDF vectors
    2. Jaccard similarity of document sets
    3. Sentence-level similarity analysis
    """
    # Preprocess documents
    stop_words = set(stopwords.words('english'))
    
    def preprocess(text):
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Tokenize each sentence into words and clean
        processed_sentences = []
        for sentence in sentences:
            # Tokenize words
            words = word_tokenize(sentence.lower())
            # Remove stopwords and non-alphabetic tokens
            words = [word for word in words if word.isalpha() and word not in stop_words]
            processed_sentences.append(words)
            
        return sentences, processed_sentences
    
    # Get original and processed sentences
    doc1_sentences, doc1_processed = preprocess(doc1)
    doc2_sentences, doc2_processed = preprocess(doc2)
    
    # 1. Cosine similarity using TF-IDF
    # Combine all sentences into documents
    doc1_text = ' '.join([' '.join(sentence) for sentence in doc1_processed])
    doc2_text = ' '.join([' '.join(sentence) for sentence in doc2_processed])
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1_text, doc2_text])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # 2. Jaccard similarity
    doc1_words = set([word for sentence in doc1_processed for word in sentence])
    doc2_words = set([word for sentence in doc2_processed for word in sentence])
    
    jaccard_sim = len(doc1_words.intersection(doc2_words)) / len(doc1_words.union(doc2_words))
    
    # 3. Sentence-level similarity analysis
    sentence_similarities = []
    
    # Compare each sentence from doc1 with each sentence from doc2
    for i, sent1 in enumerate(doc1_processed):
        for j, sent2 in enumerate(doc2_processed):
            if not sent1 or not sent2:  # Skip empty sentences
                continue
                
            # Calculate word overlap (simplified similarity)
            common_words = set(sent1).intersection(set(sent2))
            if not common_words:  # No common words
                continue
                
            similarity = len(common_words) / (len(set(sent1).union(set(sent2))))
            if similarity > 0.2:  # Threshold to filter out low similarities
                sentence_similarities.append({
                    'doc1_sentence_idx': i,
                    'doc2_sentence_idx': j,
                    'doc1_sentence': doc1_sentences[i],
                    'doc2_sentence': doc2_sentences[j],
                    'similarity': similarity
                })
    
    # Sort sentence similarities by score
    sentence_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return results
    return {
        'cosine_similarity': cosine_sim,
        'jaccard_similarity': jaccard_sim,
        'sentence_similarities': sentence_similarities[:5],  # Top 5 similar sentences
        'overall_similarity': (cosine_sim + jaccard_sim) / 2  # Average of both measures
    }

# Example usage
doc1 = """Machine learning is a field of study that gives computers the ability to learn 
without being explicitly programmed. It focuses on developing algorithms that can learn from and make 
predictions on data. Machine learning is closely related to computational statistics and mathematical optimization. 
Machine learning tasks are typically classified into several broad categories."""

doc2 = """Machine learning is a subset of artificial intelligence that provides systems the ability 
to automatically learn and improve from experience. It focuses on the development of computer programs 
that can access data and use it to learn for themselves. The primary aim is to allow computers to learn 
automatically without human intervention. Machine learning algorithms build a model based on sample data."""

# Calculate similarity
similarity_results = find_document_similarity(doc1, doc2)

# Print results
print(f"Document Similarity Analysis")
print(f"Cosine Similarity: {similarity_results['cosine_similarity']:.4f}")
print(f"Jaccard Similarity: {similarity_results['jaccard_similarity']:.4f}")
print(f"Overall Similarity: {similarity_results['overall_similarity']:.4f}")

print("\nTop Similar Sentences:")
for i, item in enumerate(similarity_results['sentence_similarities']):
    print(f"\nMatch {i+1} (Similarity: {item['similarity']:.4f}):")
    print(f"Doc1: \"{item['doc1_sentence']}\"")
    print(f"Doc2: \"{item['doc2_sentence']}\"")