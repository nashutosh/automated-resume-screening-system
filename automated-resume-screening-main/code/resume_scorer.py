import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.required_skills = set()
        self.weights = {
            'skills_match': 0.4,
            'experience_match': 0.3,
            'education_match': 0.2,
            'text_similarity': 0.1
        }
        
    def set_job_requirements(self, requirements):
        """Set job requirements for scoring"""
        self.required_skills = set(requirements.get('skills', []))
        self.required_experience = requirements.get('experience', 0)
        self.required_education = requirements.get('education', [])
        self.job_description = requirements.get('description', '')
        
    def calculate_skills_score(self, candidate_skills):
        """Calculate skills match score"""
        if not self.required_skills:
            return 0
            
        matched_skills = self.required_skills.intersection(candidate_skills)
        return len(matched_skills) / len(self.required_skills) * 100
        
    def calculate_text_similarity(self, resume_text):
        """Calculate text similarity score"""
        if not self.job_description:
            return 0
            
        # Vectorize texts
        texts = [self.job_description, resume_text]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100
        
    def get_detailed_score(self, resume_data):
        """Get detailed scoring breakdown"""
        # Calculate individual scores
        skills_score = self.calculate_skills_score(resume_data['skills'])
        text_score = self.calculate_text_similarity(resume_data['text'])
        
        # Calculate weighted total
        total_score = (
            skills_score * self.weights['skills_match'] +
            text_score * self.weights['text_similarity']
        )
        
        return {
            'total_score': total_score,
            'breakdown': {
                'skills_match': skills_score,
                'text_similarity': text_score
            },
            'matched_skills': self.required_skills.intersection(resume_data['skills'])
        } 