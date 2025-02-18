from pdfminer.high_level import extract_text
import docx2txt
import nltk
import re
import pandas as pd
import os
import PyPDF2
from nltk.tokenize import word_tokenize, sent_tokenize
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_names(txt):
    person_names = []
    for sent in nltk.sent_tokenize(txt):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_names.append(
                    ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )
    return person_names[0] if person_names else None

def extract_phone_number(resume_text):
    phone_regex = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    phone = re.findall(phone_regex, resume_text)
    if phone:
        number = ''.join(phone[0])
        if resume_text.find(number) >= 0 and len(number) < 16:
            return number
    return None

def extract_emails(resume_text):
    email_regex = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
    return re.findall(email_regex, resume_text)

def extract_education(input_text):
    school_keywords = [
        'school',
        'college',
        'university',
        'academy',
        'faculty',
        'institute',
        'diploma',
    ]
    
    organizations = []
    for sent in nltk.sent_tokenize(input_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                organizations.append(' '.join(c[0] for c in chunk.leaves()))

    education = set()
    for org in organizations:
        for word in school_keywords:
            if org.lower().find(word) >= 0:
                education.add(org)
    return education

class ResumeParser:
    def __init__(self, base_path=None):
        # Setup paths
        if base_path is None:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_path = base_path
        self.datasets_path = os.path.join(base_path, 'datasets')
        
        # Load job titles database
        self.job_titles_db = self._load_job_titles_db()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load skills database
        self.skills_db = self._load_skills_db()

    def _download_nltk_data(self):
        """Download required NLTK data packages"""
        required_packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words',
            'names',
            'stopwords'
        ]
        
        for package in required_packages:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                print(f"Error downloading NLTK package {package}: {str(e)}")

    def _load_job_titles_db(self):
        """Load job titles from CSV files"""
        try:
            # Load from job_titles_set.csv
            titles_df = pd.read_csv(os.path.join(self.datasets_path, 'job_titles_set.csv'))
            titles = set(titles_df['title'].str.lower())
            
            # Load from resumedataset.csv if exists
            resume_dataset = os.path.join(self.datasets_path, 'resumedataset.csv')
            if os.path.exists(resume_dataset):
                resume_df = pd.read_csv(resume_dataset)
                if 'Category' in resume_df.columns:
                    titles.update(resume_df['Category'].str.lower())
            
            return titles
        except Exception as e:
            print(f"Error loading job titles database: {str(e)}")
            return set()

    def _load_skills_db(self):
        """Load technical skills database"""
        skills = {
            'programming': {'python', 'java', 'c++', 'javascript', 'ruby', 'php', 'sql'},
            'frameworks': {'django', 'flask', 'react', 'angular', 'vue', 'spring'},
            'databases': {'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server'},
            'tools': {'git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp'}
        }
        return skills

    def parse_resume(self, file_path):
        """Extract text from PDF resume using pdfminer.six"""
        try:
            # Extract text from PDF
            text = extract_text(file_path)
            
            # Clean the text
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.replace('\x00', '')    # Remove null bytes
            text = text.strip()
            
            # Debug print
            print(f"\nProcessing resume: {os.path.basename(file_path)}")
            print(f"Extracted text length: {len(text)}")
            
            return text
        except Exception as e:
            print(f"Error parsing resume {file_path}: {str(e)}")
            return None

    def extract_names(self, text):
        """Extract person names from text"""
        try:
            # First look for common resume header patterns
            header_patterns = [
                r'(?i)name\s*:\s*([A-Za-z\s\.]+)',
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*\n'
            ]
            
            for pattern in header_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    name = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    name = name.strip()
                    if len(name.split()) >= 2:  # Ensure at least first and last name
                        return name
            
            # If no name found in header, try NLTK NER
            person_names = []
            sentences = nltk.sent_tokenize(text)[:3]  # Look only in first 3 sentences
            for sent in sentences:
                for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                    if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                        name = ' '.join(c[0] for c in chunk.leaves())
                        if len(name.split()) >= 2:
                            person_names.append(name)
            
            return person_names[0] if person_names else None
            
        except Exception as e:
            print(f"Error extracting names: {str(e)}")
            return None

    def extract_phone_number(self, text):
        """Extract phone number from text"""
        try:
            phone_regex = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
            phone = re.findall(phone_regex, text)
            if phone:
                number = ''.join(phone[0])
                if text.find(number) >= 0 and len(number) < 16:
                    return number
            return None
        except Exception as e:
            print(f"Error extracting phone number: {str(e)}")
            return None

    def extract_emails(self, text):
        """Extract email addresses from text"""
        try:
            email_regex = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
            return re.findall(email_regex, text.lower())
        except Exception as e:
            print(f"Error extracting emails: {str(e)}")
            return []

    def extract_education(self, text):
        """Extract education information from text"""
        try:
            education_info = []
            
            # Education section patterns
            education_headers = [
                r'education',
                r'academic background',
                r'academic qualification',
                r'educational qualification'
            ]
            
            # Find education section
            text_lower = text.lower()
            education_text = ""
            
            for header in education_headers:
                match = re.search(f"{header}.*?(?=\n\n|$)", text_lower, re.IGNORECASE | re.DOTALL)
                if match:
                    education_text = match.group()
                    break
            
            if not education_text:
                education_text = text  # Search in entire text if no section found
            
            # Degree patterns
            degree_patterns = [
                r'(?i)(?:bachelor|master|phd|doctorate)(?:\'s)?\s+(?:of|in)\s+[^.]*',
                r'(?i)b\.?(?:tech|e|sc|a|s)\.?\s+(?:in\s+)?[^.]*',
                r'(?i)m\.?(?:tech|e|sc|a|s)\.?\s+(?:in\s+)?[^.]*',
                r'(?i)ph\.?d\.?\s+(?:in\s+)?[^.]*'
            ]
            
            # Extract degrees
            for pattern in degree_patterns:
                matches = re.finditer(pattern, education_text)
                for match in matches:
                    degree = match.group().strip()
                    if len(degree) > 5:  # Avoid short matches
                        education_info.append(degree)
            
            # Extract institutions
            institution_pattern = r'(?i)(?:university|college|institute|school)\s+of\s+[^.]*'
            matches = re.finditer(institution_pattern, text)
            for match in matches:
                institution = match.group().strip()
                if len(institution) > 10:
                    education_info.append(institution)
            
            # Debug print
            if education_info:
                print("\nFound Education:")
                for edu in education_info:
                    print(f"- {edu}")
            
            return set(education_info)
            
        except Exception as e:
            print(f"Error extracting education: {str(e)}")
            return set()

    def extract_job_titles(self, text):
        """Extract job titles using pattern matching and context"""
        try:
            titles = set()
            
            # Find experience section
            experience_headers = [
                r'experience',
                r'employment history',
                r'work history',
                r'professional experience'
            ]
            
            text_lower = text.lower()
            experience_text = ""
            
            for header in experience_headers:
                match = re.search(f"{header}.*?(?=\n\n|$)", text_lower, re.IGNORECASE | re.DOTALL)
                if match:
                    experience_text = match.group()
                    break
            
            if not experience_text:
                experience_text = text
            
            # Job title patterns
            job_patterns = [
                r'(?i)(?:senior|lead|principal|staff|junior|associate)?\s*(?:software|data|full[\s-]stack|front[\s-]end|back[\s-]end|devops|ml|ai|cloud|systems|application|mobile|web)?\s*(?:engineer|developer|scientist|architect|analyst|consultant|specialist)',
                r'(?i)(?:project|product|program|technical|engineering|development|team)?\s*(?:manager|lead|director|head)',
                r'(?i)(?:director|vp|head|chief|cto|ceo|cio|cfo|coo)',
                r'(?i)(?:business|systems|data|financial|marketing)\s*(?:analyst|consultant)',
                r'(?i)(?:research|teaching|graduate)\s*(?:assistant|associate|fellow)',
                r'(?i)(?:attorney|lawyer|counsel|paralegal)',
                r'(?i)(?:nurse|rn|lpn|nurse practitioner|clinical nurse)',
                r'(?i)(?:sales|account)\s*(?:representative|manager|executive)'
            ]
            
            # Extract titles
            for pattern in job_patterns:
                matches = re.finditer(pattern, experience_text)
                for match in matches:
                    title = match.group().strip()
                    if title and len(title) > 3:
                        titles.add(title)
            
            # Debug print
            if titles:
                print("\nFound Job Titles:")
                for title in titles:
                    print(f"- {title}")
            
            return list(titles)
            
        except Exception as e:
            print(f"Error extracting job titles: {str(e)}")
            return []

    def extract_job_titles_from_db(self, text, job_title_db):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(text)

        filtered_tokens = [w for w in word_tokens if w not in stop_words]
        filtered_tokens = [w for w in filtered_tokens if w.isalpha()]

        grams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

        found_skills = set()

        for i in filtered_tokens:
            if i.lower() in job_title_db:
                found_skills.add(i)

        for i in grams:
            if i.lower() in job_title_db:
                found_skills.add(i)

        return found_skills

    def extract_skills(self, text):
        """Extract technical skills from resume"""
        try:
            found_skills = {category: set() for category in self.skills_db}
            
            # Convert text to lowercase for matching
            text_lower = text.lower()
            
            # Look for skills in each category
            for category, skills in self.skills_db.items():
                for skill in skills:
                    if skill in text_lower:
                        found_skills[category].add(skill)
            
            # Debug print
            if any(skills for skills in found_skills.values()):
                print("\nFound Skills:")
                for category, skills in found_skills.items():
                    if skills:
                        print(f"{category.title()}:")
                        for skill in skills:
                            print(f"- {skill}")
            
            return found_skills
        except Exception as e:
            print(f"Error extracting skills: {str(e)}")
            return {} 