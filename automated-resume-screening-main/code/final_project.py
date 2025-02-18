from os import path
from glob import glob  
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

class ResumeScreener:
    def __init__(self):
        """Initialize the resume screener with required NLTK data."""
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('stopwords')

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file."""
        return extract_text(pdf_path)

    def extract_names(self, txt):
        """Extract person names from text using NLTK."""
        person_names = []
        for sent in nltk.sent_tokenize(txt):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                    person_names.append(
                        ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                    )
        return person_names[0] if person_names else None

    def extract_phone_number(self, resume_text):
        """Extract phone number from text using regex."""
        phone_regex = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        phone = re.findall(phone_regex, resume_text)
        if phone:
            number = ''.join(phone[0])
            if resume_text.find(number) >= 0 and len(number) < 16:
                return number
        return None

    def extract_emails(self, resume_text):
        """Extract email addresses from text using regex."""
        email_regex = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
        return re.findall(email_regex, resume_text)

    def extract_education(self, input_text):
        """Extract education information from text."""
        RESERVED_WORDS = [
            'school', 'college', 'univers', 'academy', 'faculty', 'institute',
            'faculdades', 'Schola', 'schule', 'lise', 'lyceum', 'lycee',
            'polytechnic', 'kolej', 'ünivers', 'okul',
        ]
        
        organizations = []
        for sent in nltk.sent_tokenize(input_text):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                    organizations.append(' '.join(c[0] for c in chunk.leaves()))

        education = set()
        for org in organizations:
            for word in RESERVED_WORDS:
                if org.lower().find(word) >= 0:
                    education.add(org)
        return education

    def extract_job_titles(self, input_text, job_titles_db):
        """Extract job titles from text."""
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(input_text)

        filtered_tokens = [w for w in word_tokens if w not in stop_words]
        filtered_tokens = [w for w in word_tokens if w.isalpha()]

        grams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

        found_skills = set()

        for token in filtered_tokens:
            if token.lower() in job_titles_db:
                found_skills.add(token)

        for ngram in grams:
            if ngram.lower() in job_titles_db:
                found_skills.add(ngram)

        return found_skills

    def process_resumes(self, resume_dir, job_titles_csv):
        """Process all resumes in the directory."""
        # Find all PDF files
        resumepaths = glob(path.join(resume_dir, "*.pdf"))
        
        # Create initial dataframe
        df = pd.DataFrame(resumepaths, columns=['path'])
        
        # Extract text from PDFs
        df['text'] = df['path'].apply(self.extract_text_from_pdf)
        
        # Extract information
        df['name'] = df.text.apply(self.extract_names)
        df['phone'] = df.text.apply(self.extract_phone_number)
        df['email'] = df.text.apply(self.extract_emails)
        df['school'] = df.text.apply(self.extract_education)
        
        # Load job titles and extract them
        job_titles_db = pd.read_csv(job_titles_csv).title.values
        df['job_titles'] = df.text.apply(lambda x: self.extract_job_titles(x, job_titles_db))
        
        return df

    def calculate_similarities(self, df, job_description):
        """Calculate similarity between resumes and job description."""
        # Add job description to dataframe
        new_row = pd.DataFrame({'path':'job_description', 'text': job_description}, index=[0])
        df = pd.concat([new_row, df]).reset_index(drop=True)
        
        # Clean text
        stop_words_l = stopwords.words('english')
        df['text_cleaned'] = df.text.apply(
            lambda x: " ".join(
                re.sub(r'[^a-zA-Z]',' ',w).lower() 
                for w in x.split() 
                if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l
            )
        )
        
        # Calculate TF-IDF and similarities
        tfidfvectoriser = TfidfVectorizer()
        tfidfvectoriser.fit(df.text_cleaned)
        tfidf_vectors = tfidfvectoriser.transform(df.text_cleaned)
        
        similarities = np.dot(tfidf_vectors, tfidf_vectors.T).toarray()
        
        for i in range(len(similarities[0])):
            df.loc[i, "similarity"] = similarities[0][i]
            
        df.sort_values(by='similarity', ascending=False, inplace=True)
        
        # Remove job description and reset index
        df = df.drop(0)
        df.reset_index(drop=True, inplace=True)
        
        return df[['path', 'name', 'email', 'similarity']]

def main():
    # Initialize screener
    screener = ResumeScreener()
    
    print("Processing resumes...")
    # Process resumes
    df = screener.process_resumes(
        "datasets/resumes-list",
        "datasets/job_titles_set.csv"
    )
    
    print("Loading job description...")
    # Load job description
    with open("datasets/job_description.txt", "r") as f:
        job_description = f.read()
    
    print("Calculating similarities...")
    # Calculate similarities and get results
    results = screener.calculate_similarities(df, job_description)
    
    # Display results
    print("\nRanked Results:")
    print(results)
    
    # Save results
    results.to_csv("screening_results.csv", index=False)
    print("\nResults saved to screening_results.csv")

if __name__ == "__main__":
    main() 