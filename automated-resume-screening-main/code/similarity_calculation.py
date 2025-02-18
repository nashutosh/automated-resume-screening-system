import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def load_sample_documents():
    """Load sample documents for similarity comparison."""
    return [
        'Machine learning is the study of computer algorithms that improve automatically through experience.\
        Machine learning algorithms build a mathematical model based on sample data, known as training data.\
        The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
        where no fully satisfactory algorithm is available.',
        
        'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
        The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
        
        'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
        It involves computers learning from data provided so that they carry out certain tasks.',
        
        'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal"\
        or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
        
        'Software engineering is the systematic application of engineering approaches to the development of software.\
        Software engineering is a computing discipline.',
        
        'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
        about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
        Developing a machine learning application is more iterative and explorative process than software engineering.'
    ]

def clean_documents(documents_df):
    """Clean documents by removing special characters and stop words."""
    stop_words_l = stopwords.words('english')
    documents_df['documents_cleaned'] = documents_df.documents.apply(
        lambda x: " ".join(
            re.sub(r'[^a-zA-Z]',' ',w).lower() 
            for w in x.split() 
            if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l
        )
    )
    return documents_df

def calculate_similarities(documents_df):
    """Calculate TF-IDF vectors and similarity matrices."""
    tfidfvectoriser = TfidfVectorizer()
    tfidfvectoriser.fit(documents_df.documents_cleaned)
    tfidf_vectors = tfidfvectoriser.transform(documents_df.documents_cleaned)
    
    pairwise_similarities = np.dot(tfidf_vectors, tfidf_vectors.T).toarray()
    pairwise_differences = euclidean_distances(tfidf_vectors)
    
    return pairwise_similarities, pairwise_differences

def most_similar(doc_id, similarity_matrix, matrix, documents_df):
    """Find and print most similar documents to the given document."""
    print(f'Document: {documents_df.iloc[doc_id]["documents"]}')
    print('\n')
    print('Similar Documents:')
    
    if matrix == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix == 'Euclidean Distance':
        similar_ix = np.argsort(similarity_matrix[doc_id])
        
    for ix in similar_ix:
        if ix == doc_id:
            continue
        print('\n')
        print(f'Document: {documents_df.iloc[ix]["documents"]}')
        print(f'{matrix} : {similarity_matrix[doc_id][ix]}')

def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts"""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity) * 100, 2)
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0

def main():
    # Download required NLTK data
    nltk.download('stopwords')
    
    # Load sample documents
    documents = load_sample_documents()
    
    # Create DataFrame
    documents_df = pd.DataFrame(documents, columns=['documents'])
    
    # Clean documents
    documents_df = clean_documents(documents_df)
    
    # Calculate similarities
    pairwise_similarities, pairwise_differences = calculate_similarities(documents_df)
    
    # Print similar documents using cosine similarity
    print("Using Cosine Similarity:")
    most_similar(3, pairwise_similarities, 'Cosine Similarity', documents_df)
    
    # Uncomment to print similar documents using euclidean distance
    # print("\nUsing Euclidean Distance:")
    # most_similar(3, pairwise_differences, 'Euclidean Distance', documents_df)

if __name__ == "__main__":
    main() 