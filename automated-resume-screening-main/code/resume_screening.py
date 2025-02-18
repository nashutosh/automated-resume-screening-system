import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re

def preprocess_text(text):
    """Preprocess text by removing special characters and stopwords."""
    if isinstance(text, str):  # Add check for string type
        stop_words = stopwords.words('english')
        return " ".join(
            re.sub(r'[^a-zA-Z]',' ',w).lower() 
            for w in text.split() 
            if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words
        )
    return ""  # Return empty string for non-string inputs

def calculate_similarities(df):
    """Calculate TF-IDF similarities between documents."""
    tfidfvectoriser = TfidfVectorizer()
    tfidfvectoriser.fit(df.text_cleaned)
    tfidf_vectors = tfidfvectoriser.transform(df.text_cleaned)
    
    similarities = np.dot(tfidf_vectors, tfidf_vectors.T).toarray()
    
    for i in range(len(similarities[0])):
        df.loc[i, "similarity"] = similarities[0][i]
        
    df.sort_values(by='similarity', ascending=False, inplace=True)
    
    return df

def screen_resumes(resumes_df, job_description):
    """Screen resumes against a job description using TF-IDF similarity."""
    # Add job description as first row
    new_row = pd.DataFrame({
        'path':'job_description', 
        'text': job_description
    }, index=[0])
    
    df = pd.concat([new_row, resumes_df]).reset_index(drop=True)
    
    # Clean text
    df['text_cleaned'] = df.text.apply(preprocess_text)
    
    # Calculate similarities
    df = calculate_similarities(df)
    
    # Remove job description row and reset index
    df = df.drop(0)
    df.reset_index(drop=True, inplace=True)
    
    return df[['path', 'name', 'email', 'similarity']] 