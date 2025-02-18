import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd
import os
import json
from resume_parser import ResumeParser
from job_title_analysis import JobTitleAnalyzer
from similarity_calculation import calculate_similarity
import numpy as np

class ResumeScreeningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Screening System")
        self.root.geometry("1600x900")
        
        # Set theme and colors
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'accent': '#3498db',
            'background': '#ecf0f1',
            'text': '#2c3e50'
        }
        
        # Configure styles
        self.style.configure('Treeview', 
                           background=self.colors['background'],
                           fieldbackground=self.colors['background'],
                           foreground=self.colors['text'])
        
        self.style.configure('TFrame', background=self.colors['background'])
        self.style.configure('TLabel', background=self.colors['background'],
                           foreground=self.colors['text'])
        self.style.configure('TButton', 
                           background=self.colors['accent'],
                           foreground='white')
        
        # Initialize components
        self.setup_system()
        self.create_gui()
        
    def setup_system(self):
        """Initialize the resume screening system"""
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.datasets_path = os.path.join(self.base_path, 'datasets')
        self.resumes_path = os.path.join(self.datasets_path, 'resumes-list')
        
        self.resume_parser = ResumeParser()
        self.job_title_analyzer = JobTitleAnalyzer(
            os.path.join(self.datasets_path, 'job_titles_set.csv')
        )
        
        self.results = []
        
    def create_gui(self):
        """Create the GUI layout"""
        # Configure root
        self.root.configure(bg=self.colors['background'])
        
        # Create main frames
        self.create_frames()

    def create_frames(self):
        """Create the main frame layout"""
        # Main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create paned window for split layout
        paned = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        left_frame = ttk.Frame(paned, padding="10")
        
        # Title
        title_label = ttk.Label(
            left_frame, 
            text="Resume Screening System",
            font=('Helvetica', 14, 'bold'),
            foreground=self.colors['primary']
        )
        title_label.pack(pady=(0, 20))
        
        # Process button
        self.process_btn = ttk.Button(
            left_frame, 
            text="Process Resumes",
            command=self.process_resumes
        )
        self.process_btn.pack(pady=5, fill=tk.X)
        
        # Job description
        ttk.Label(
            left_frame, 
            text="Job Description:",
            font=('Helvetica', 10, 'bold')
        ).pack(pady=5)
        
        self.job_desc_text = scrolledtext.ScrolledText(
            left_frame, 
            height=15, 
            width=40,
            font=('Helvetica', 10),
            bg='white'
        )
        self.job_desc_text.pack(pady=5)
        
        # Load job description
        self.load_job_description()
        
        # Dashboard button
        self.dashboard_btn = ttk.Button(
            left_frame,
            text="Open Analysis Dashboard",
            command=self.open_analysis_dashboard
        )
        self.dashboard_btn.pack(pady=5, fill=tk.X)
        
        # Initially disable the dashboard button
        self.dashboard_btn.configure(state='disabled')
        
        # Right panel for results
        right_frame = ttk.Frame(paned, padding="10")
        
        # Add frames to paned window
        paned.add(left_frame, weight=1)
        paned.add(right_frame, weight=3)
        
        # Create results view in right frame
        self.create_results_view(right_frame)

    def create_results_view(self, parent):
        """Create the results table view"""
        # Results frame
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            results_frame,
            text="Resume Analysis Results",
            font=('Helvetica', 12, 'bold'),
            foreground=self.colors['primary']
        ).pack(pady=(0, 10))
        
        # Create table
        columns = ('Name', 'Email', 'Phone', 'Match %', 'Top Skills', 'Education')
        self.tree = ttk.Treeview(
            results_frame, 
            columns=columns, 
            show='headings',
            height=10
        )
        
        # Configure columns
        column_widths = {
            'Name': 150,
            'Email': 200,
            'Phone': 120,
            'Match %': 80,
            'Top Skills': 250,
            'Education': 250
        }
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col])
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(
            results_frame, 
            orient=tk.VERTICAL, 
            command=self.tree.yview
        )
        x_scrollbar = ttk.Scrollbar(
            results_frame, 
            orient=tk.HORIZONTAL, 
            command=self.tree.xview
        )
        
        self.tree.configure(
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set
        )
        
        # Pack components
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.show_resume_details)

    def create_visualizations(self):
        """Create visualization area"""
        # Create container frame with padding
        viz_container = ttk.Frame(self.bottom_frame, padding="20")
        viz_container.pack(fill=tk.BOTH, expand=True)
        
        # Title with better spacing
        ttk.Label(
            viz_container,
            text="Quick Overview",
            font=('Helvetica', 14, 'bold'),
            foreground=self.colors['primary']
        ).pack(pady=(0, 20))
        
        # Summary metrics in a row
        metrics_frame = ttk.Frame(viz_container)
        metrics_frame.pack(fill=tk.X, pady=(0, 20))
        
        if self.results:
            metrics = [
                ("Total Resumes", len(self.results)),
                ("Average Match", f"{sum(r['similarity_score'] for r in self.results) / len(self.results):.1f}%"),
                ("Top Match", f"{max(r['similarity_score'] for r in self.results):.1f}%")
            ]
            
            for label, value in metrics:
                metric_frame = ttk.Frame(metrics_frame, padding="10")
                metric_frame.pack(side=tk.LEFT, expand=True)
                
                ttk.Label(
                    metric_frame,
                    text=label,
                    font=('Helvetica', 10)
                ).pack()
                
                ttk.Label(
                    metric_frame,
                    text=str(value),
                    font=('Helvetica', 12, 'bold'),
                    foreground=self.colors['accent']
                ).pack()
        
    def create_advanced_visualizations(self):
        """Create advanced visualization tabs"""
        # Create notebook for multiple visualizations
        self.viz_notebook = ttk.Notebook(self.bottom_frame)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Overview tab
        overview_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(overview_frame, text='Overview')
        
        # Create figure with subplots
        self.fig = plt.Figure(figsize=(15, 6))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=overview_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Skills Analysis tab
        skills_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(skills_frame, text='Skills Analysis')
        
        self.skills_fig = plt.Figure(figsize=(15, 6))
        self.skills_ax = self.skills_fig.add_subplot(111)
        
        self.skills_canvas = FigureCanvasTkAgg(self.skills_fig, master=skills_frame)
        self.skills_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_job_description(self):
        """Load job description from file"""
        try:
            job_desc_path = os.path.join(self.datasets_path, 'job_description.txt')
            with open(job_desc_path, 'r', encoding='utf-8') as file:
                self.job_desc_text.insert(tk.END, file.read())
        except Exception as e:
            print(f"Error loading job description: {str(e)}")
            
    def process_resumes(self):
        """Process all resumes and update GUI"""
        self.results = []
        self.tree.delete(*self.tree.get_children())
        
        # Get job description
        job_description = self.job_desc_text.get("1.0", tk.END).strip()
        
        if not job_description:
            messagebox.showwarning(
                "Missing Input",
                "Please enter a job description before processing resumes."
            )
            return
        
        try:
            # Process each resume
            for filename in os.listdir(self.resumes_path):
                if filename.lower().endswith(('.pdf', '.docx', '.doc')):
                    resume_path = os.path.join(self.resumes_path, filename)
                    result = self.process_single_resume(resume_path, job_description)
                    if result:
                        self.results.append(result)
            
            # Sort results by similarity score
            self.results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Update table
            for result in self.results:
                skills = ', '.join([skill for category in result['skills'].values() 
                                  for skill in category])
                self.tree.insert('', tk.END, values=(
                    result['name'],
                    result['email'],
                    result['phone'],
                    f"{result['similarity_score']:.1f}%",
                    skills[:50] + '...' if len(skills) > 50 else skills,
                    result['education'][0] if result['education'] else 'Not specified'
                ))
            
            # Enable the dashboard button after processing
            self.dashboard_btn.config(state=tk.NORMAL)
            
            # Show success message
            messagebox.showinfo(
                "Processing Complete",
                f"Successfully processed {len(self.results)} resumes."
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"An error occurred while processing resumes: {str(e)}"
            )
            
    def process_single_resume(self, resume_path, job_description):
        """Process a single resume"""
        try:
            resume_text = self.resume_parser.parse_resume(resume_path)
            if not resume_text:
                return None

            # Extract information
            name = self.resume_parser.extract_names(resume_text)
            email = self.resume_parser.extract_emails(resume_text)
            phone = self.resume_parser.extract_phone_number(resume_text)
            education = self.resume_parser.extract_education(resume_text)
            job_titles = self.resume_parser.extract_job_titles(resume_text)
            skills = self.resume_parser.extract_skills(resume_text)

            # Calculate similarity
            similarity_score = calculate_similarity(resume_text, job_description)

            return {
                'resume_path': resume_path,
                'name': name,
                'email': email[0] if email else None,
                'phone': phone,
                'education': list(education),
                'job_titles': job_titles,
                'skills': skills,
                'similarity_score': similarity_score
            }
            
        except Exception as e:
            print(f"Error processing resume {resume_path}: {str(e)}")
            return None
            
    def update_visualizations(self):
        """Update visualization graphs"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Similarity score distribution
        scores = [r['similarity_score'] for r in self.results]
        
        # Plot histogram with custom style
        n, bins, patches = self.ax1.hist(scores, bins=10, color=self.colors['accent'], alpha=0.7)
        self.ax1.set_title('Match Score Distribution', pad=20, fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Match Score (%)', fontsize=10)
        self.ax1.set_ylabel('Number of Resumes', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        
        # Add mean line
        if scores:
            mean_score = sum(scores) / len(scores)
            self.ax1.axvline(mean_score, color='red', linestyle='dashed', alpha=0.8)
            self.ax1.text(mean_score + 2, self.ax1.get_ylim()[1]*0.9, 
                         f'Mean: {mean_score:.1f}%', 
                         color='red')
        
        # Skills distribution
        all_skills = []
        for result in self.results:
            for category, skills in result['skills'].items():
                all_skills.extend(skills)
        
        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts()[:10]
            
            # Plot bar chart
            bars = self.ax2.bar(range(len(skill_counts)), 
                               skill_counts.values,
                               color=self.colors['accent'],
                               alpha=0.7)
            
            # Customize the plot
            self.ax2.set_title('Top 10 Skills', pad=20, fontsize=12, fontweight='bold')
            self.ax2.set_xlabel('Skills', fontsize=10)
            self.ax2.set_ylabel('Frequency', fontsize=10)
            self.ax2.set_xticks(range(len(skill_counts)))
            self.ax2.set_xticklabels(skill_counts.index, rotation=45, ha='right')
            self.ax2.grid(True, alpha=0.3)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                self.ax2.text(bar.get_x() + bar.get_width()/2., height,
                             f'{int(height)}',
                             ha='center', va='bottom')
        
        # Adjust layout to prevent overlapping
        self.fig.tight_layout()
        self.canvas.draw()
        
    def show_resume_details(self, event):
        """Show detailed information for selected resume"""
        selection = self.tree.selection()
        if not selection:
            return
            
        # Get selected result
        idx = self.tree.index(selection[0])
        result = self.results[idx]
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Resume Details")
        popup.geometry("800x600")
        popup.config(bg=self.colors['background'])
        
        # Create text widget
        text = scrolledtext.ScrolledText(
            popup, 
            wrap=tk.WORD, 
            width=80, 
            height=30,
            font=('Helvetica', 10),
            bg='white'
        )
        text.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Insert details with formatting
        text.tag_configure('heading', 
                          font=('Helvetica', 12, 'bold'),
                          foreground=self.colors['primary'])
        text.tag_configure('subheading', 
                          font=('Helvetica', 10, 'bold'),
                          foreground=self.colors['secondary'])
        
        # Insert details
        text.insert(tk.END, "Resume Details\n\n", 'heading')
        
        details = f"""
File: {os.path.basename(result['resume_path'])}
Name: {result['name'] or 'Not found'}
Email: {result['email'] or 'Not found'}
Phone: {result['phone'] or 'Not found'}
Match Score: {result['similarity_score']:.1f}%\n
"""
        text.insert(tk.END, details)
        
        # Education section
        text.insert(tk.END, "\nEducation:\n", 'subheading')
        if result['education']:
            for edu in result['education']:
                text.insert(tk.END, f"• {edu}\n")
        else:
            text.insert(tk.END, "No education information found\n")
        
        # Job Titles section
        text.insert(tk.END, "\nJob Titles:\n", 'subheading')
        if result['job_titles']:
            for title in result['job_titles']:
                text.insert(tk.END, f"• {title}\n")
        else:
            text.insert(tk.END, "No job titles found\n")
        
        # Skills section
        text.insert(tk.END, "\nSkills:\n", 'subheading')
        for category, skills in result['skills'].items():
            if skills:
                text.insert(tk.END, f"\n{category.title()}:\n")
                for skill in skills:
                    text.insert(tk.END, f"• {skill}\n")
        
        text.configure(state='disabled')

    def open_analysis_dashboard(self):
        """Open the analysis dashboard"""
        if not self.results:
            messagebox.showwarning(
                "No Data",
                "Please process resumes first before opening the dashboard."
            )
            return
        
        # Check if dashboard already exists
        if hasattr(self, 'dashboard') and self.dashboard.window.winfo_exists():
            self.dashboard.window.lift()  # Bring existing window to front
            return
        
        try:
            # Ensure all results have the required fields
            for result in self.results:
                if not isinstance(result.get('skills', {}), dict):
                    result['skills'] = {}
                
                result.setdefault('name', None)
                result.setdefault('email', None)
                result.setdefault('phone', None)
                result.setdefault('education', [])
                result.setdefault('job_titles', [])
                result.setdefault('similarity_score', 0.0)
            
            from analysis_dashboard import AnalysisDashboard
            self.dashboard = AnalysisDashboard(self.root, self.results, self.colors)
            
            # Configure window behavior
            self.dashboard.window.transient(self.root)  # Set as child window
            self.dashboard.window.grab_set()  # Make window modal
            
            # Handle window close
            self.dashboard.window.protocol("WM_DELETE_WINDOW", self.on_dashboard_close)
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"An error occurred while opening the dashboard: {str(e)}"
            )

    def on_dashboard_close(self):
        """Handle dashboard window closing"""
        if hasattr(self, 'dashboard'):
            self.dashboard.window.grab_release()  # Release modal state
            self.dashboard.window.destroy()
            del self.dashboard

def main():
    root = tk.Tk()
    app = ResumeScreeningApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 