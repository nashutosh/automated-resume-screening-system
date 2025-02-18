import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import os

class JobTitleAnalyzer:
    def __init__(self, dataset_path='datasets/job_titles_set.csv'):
        self.stop_words = set(stopwords.words("english"))
        self.dataset_path = dataset_path
        self.responsibilities_classifier = None
        self.departments_classifier = None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

    def load_and_train_classifiers(self):
        """Load job titles from CSV and train the classifiers"""
        try:
            df = pd.read_csv(self.dataset_path)
            raw_job_titles = []

            for _, row in df.iterrows():
                title = row['title']
                related_titles = str(row['top related titles']).split(',')
                
                # Determine responsibility and department
                responsibility, department = self._categorize_job(related_titles)
                raw_job_titles.append({
                    "title": title,
                    "responsibility": responsibility,
                    "department": department
                })

            # Train classifiers
            self._train_classifiers(raw_job_titles)
            return True
        except Exception as e:
            print(f"Error loading job titles: {str(e)}")
            return False

    def _categorize_job(self, related_titles):
        """Categorize job based on related titles"""
        related_text = ' '.join(related_titles).lower()
        
        if any(word in related_text for word in ['technical', 'engineer', 'developer']):
            return "Technical", "Engineering"
        elif any(word in related_text for word in ['manage', 'director', 'lead']):
            return "Management", "Management"
        elif any(word in related_text for word in ['sales', 'account']):
            return "Sales", "Sales"
        return "Other", "Other"

    def _train_classifiers(self, raw_job_titles):
        """Train the classifiers with the job titles data"""
        responsibility_features = [
            (self.get_title_features(title['title']), title['responsibility']) 
            for title in raw_job_titles
        ]
        department_features = [
            (self.get_title_features(title['title']), title['department']) 
            for title in raw_job_titles
        ]
        
        self.responsibilities_classifier = nltk.NaiveBayesClassifier.train(responsibility_features)
        self.departments_classifier = nltk.NaiveBayesClassifier.train(department_features)

    def get_title_features(self, title):
        """Extract features from a job title"""
        features = {}
        word_tokens = nltk.word_tokenize(title)
        filtered_words = [w for w in word_tokens if w not in self.stop_words]
        
        for word in filtered_words:
            features[f'contains({word.lower()})'] = True
        
        if filtered_words:
            features[f'first({filtered_words[0].lower()})'] = True
            features[f'last({filtered_words[-1].lower()})'] = True
        
        return features

    def analyze_title(self, title):
        """Analyze a job title and return its classifications with confidence scores"""
        if not self.responsibilities_classifier or not self.departments_classifier:
            if not self.load_and_train_classifiers():
                return None

        features = self.get_title_features(title)
        
        # Get classifications
        responsibility = self.responsibilities_classifier.classify(features)
        department = self.departments_classifier.classify(features)
        
        # Get confidence scores
        resp_prob = self.responsibilities_classifier.prob_classify(features)
        dept_prob = self.departments_classifier.prob_classify(features)
        
        return {
            'title': title,
            'responsibility': responsibility,
            'department': department,
            'responsibility_confidence': round(100 * resp_prob.prob(resp_prob.max())),
            'department_confidence': round(100 * dept_prob.prob(dept_prob.max()))
        }

    def get_first_title(self, title):
        """Extract the first title from a compound title"""
        title = re.sub(r"[Cc]o[\-\ ]","", title)
        split_titles = re.split(r"\,|\-|\||\&|\:|\/|and", title)
        return split_titles[0].strip()