import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import platform

class AnalysisDashboard:
    def __init__(self, parent, results, colors):
        self.window = tk.Toplevel(parent)
        self.window.title("Resume Analysis Dashboard")
        
        # Get screen dimensions and set window size to 90%
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.85)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Configure window
        self.window.configure(bg=colors['background'])
        self.results = results
        self.colors = colors
        
        # Configure styles for better appearance
        style = ttk.Style()
        style.configure('Dashboard.TNotebook', 
                       background=colors['background'],
                       tabmargins=[2, 5, 2, 0])
        style.configure('Dashboard.TFrame', 
                       background=colors['background'])
        style.configure('Card.TFrame',
                       background='white',
                       relief='solid',
                       borderwidth=1)
        
        # Create notebook with custom style
        self.notebook = ttk.Notebook(self.window, style='Dashboard.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_overview_tab()
        self.create_skills_tab()
        self.create_comparison_tab()
        self.create_recommendations_tab()
        self.create_email_tab()
        
    def create_scrollable_container(self, parent):
        """Create a scrollable container for tab content"""
        # Create container frame
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(container, bg=self.colors['background'])
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Create main frame inside canvas
        scrollable_frame = ttk.Frame(canvas, style='Dashboard.TFrame')
        
        # Configure scrolling
        def on_frame_configure(event):
            """Reset the scroll region to encompass the inner frame"""
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def on_canvas_configure(event):
            """When canvas is resized, resize the inner frame to match"""
            canvas.itemconfig(frame_id, width=event.width)
        
        def on_mousewheel(event):
            """Handle mouse wheel scrolling"""
            if canvas.winfo_height() < scrollable_frame.winfo_height():
                canvas.yview_scroll(int(-1 * (event.delta/120)), "units")
        
        def on_enter_widget(event):
            """Bind mousewheel when mouse enters the widget"""
            if platform.system() == 'Windows':
                canvas.bind_all("<MouseWheel>", on_mousewheel)
            else:
                canvas.bind_all("<Button-4>", on_mousewheel)
                canvas.bind_all("<Button-5>", on_mousewheel)
        
        def on_leave_widget(event):
            """Unbind mousewheel when mouse leaves the widget"""
            if platform.system() == 'Windows':
                canvas.unbind_all("<MouseWheel>")
            else:
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
        
        # Bind events
        scrollable_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        
        # Bind mouse enter/leave events for proper scrolling
        canvas.bind('<Enter>', on_enter_widget)
        canvas.bind('<Leave>', on_leave_widget)
        
        # Create window in canvas
        frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrolling components with padding
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        return scrollable_frame
        
    def create_overview_tab(self):
        """Create enhanced overview dashboard tab"""
        overview_tab = ttk.Frame(self.notebook, style='Dashboard.TFrame')
        self.notebook.add(overview_tab, text='Overview')
        
        content_frame = self.create_scrollable_container(overview_tab)
        
        # Top metrics row with enhanced styling
        metrics_frame = ttk.Frame(content_frame, style='Dashboard.TFrame')
        metrics_frame.pack(fill=tk.X, pady=10, padx=20)
        
        # Calculate metrics
        avg_score = sum(r['similarity_score'] for r in self.results) / len(self.results)
        max_score = max(r['similarity_score'] for r in self.results)
        total_skills = len(set(
            skill for r in self.results 
            for skills in r['skills'].values() 
            for skill in skills
        ))
        
        # Create enhanced metric cards
        metrics = [
            ('Total Resumes', len(self.results), '📄'),
            ('Average Match', f"{avg_score:.1f}%", '📊'),
            ('Top Match', f"{max_score:.1f}%", '🏆'),
            ('Total Skills', total_skills, '🔧')
        ]
        
        for icon, title, value in metrics:
            card = ttk.Frame(metrics_frame, style='Card.TFrame')
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
            
            # Icon
            ttk.Label(
                card,
                text=icon,
                font=('Helvetica', 24)
            ).pack(pady=(10, 5))
            
            # Title
            ttk.Label(
                card,
                text=title,
                font=('Helvetica', 10)
            ).pack()
            
            # Value
            ttk.Label(
                card,
                text=str(value),
                font=('Helvetica', 14, 'bold'),
                foreground=self.colors['accent']
            ).pack(pady=(0, 10))
        
        # Charts with better sizing
        charts_frame = ttk.Frame(content_frame, style='Dashboard.TFrame')
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)
        
        fig = Figure(figsize=(12, 6), dpi=100)
        fig.set_facecolor(self.colors['background'])
        
        # Enhanced score distribution
        ax1 = fig.add_subplot(121)
        scores = [r['similarity_score'] for r in self.results]
        n, bins, patches = ax1.hist(scores, bins=10, color=self.colors['accent'], alpha=0.7)
        
        # Add mean line
        mean_score = np.mean(scores)
        ax1.axvline(mean_score, color='red', linestyle='--', alpha=0.8)
        ax1.text(mean_score + 2, ax1.get_ylim()[1]*0.9, 
                 f'Mean: {mean_score:.1f}%', 
                 color='red')
        
        ax1.set_title('Match Score Distribution', pad=20)
        ax1.set_xlabel('Match Score (%)')
        ax1.set_ylabel('Number of Resumes')
        ax1.grid(True, alpha=0.3)
        
        # Enhanced top candidates visualization
        ax2 = fig.add_subplot(122)
        top_candidates = sorted(
            self.results,
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:5]
        
        names = [r['name'] or f'Candidate {i+1}' for i, r in enumerate(top_candidates)]
        scores = [r['similarity_score'] for r in top_candidates]
        
        bars = ax2.barh(range(len(names)), scores, 
                        color=self.colors['accent'],
                        alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    ha='left', va='center')
        
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_title('Top 5 Candidates')
        ax2.set_xlabel('Match Score (%)')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_skills_tab(self):
        """Create enhanced skills analysis tab"""
        skills_tab = ttk.Frame(self.notebook, style='Dashboard.TFrame')
        self.notebook.add(skills_tab, text='Skills Analysis')
        
        content_frame = self.create_scrollable_container(skills_tab)
        
        # Create main sections
        top_frame = ttk.Frame(content_frame, style='Dashboard.TFrame')
        top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Skills summary cards
        summary_frame = ttk.Frame(top_frame, style='Dashboard.TFrame')
        summary_frame.pack(fill=tk.X, pady=10)
        
        # Calculate skills metrics
        all_skills = []
        category_skills = {}
        for result in self.results:
            for category, skills in result['skills'].items():
                if category not in category_skills:
                    category_skills[category] = set()
                category_skills[category].update(skills)
                all_skills.extend(skills)
        
        unique_skills = len(set(all_skills))
        avg_skills_per_candidate = len(all_skills) / len(self.results)
        
        # Create summary cards
        metrics = [
            ('Total Unique Skills', unique_skills, '🔍'),
            ('Skill Categories', len(category_skills), '📑'),
            ('Avg Skills/Candidate', f"{avg_skills_per_candidate:.1f}", '📊'),
            ('Most Common Category', max(category_skills.keys(), key=lambda k: len(category_skills[k])), '🏆')
        ]
        
        for icon, title, value in metrics:
            card = ttk.Frame(summary_frame, style='Card.TFrame')
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
            
            ttk.Label(card, text=icon, font=('Helvetica', 24)).pack(pady=(10, 5))
            ttk.Label(card, text=title, font=('Helvetica', 10)).pack()
            ttk.Label(
                card,
                text=str(value),
                font=('Helvetica', 14, 'bold'),
                foreground=self.colors['accent']
            ).pack(pady=(0, 10))
        
        # Create charts
        charts_frame = ttk.Frame(content_frame, style='Dashboard.TFrame')
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.set_facecolor(self.colors['background'])
        
        # Skills distribution by category
        ax1 = fig.add_subplot(121)
        categories = list(category_skills.keys())
        sizes = [len(skills) for skills in category_skills.values()]
        
        patches, texts, autotexts = ax1.pie(
            sizes,
            labels=categories,
            autopct='%1.1f%%',
            colors=plt.cm.Pastel1(np.linspace(0, 1, len(categories))),
            startangle=90
        )
        
        # Make percentage labels easier to read
        plt.setp(autotexts, size=9, weight="bold")
        plt.setp(texts, size=9)
        
        ax1.set_title('Skills Distribution by Category', pad=20, fontsize=12)
        
        # Top skills bar chart
        ax2 = fig.add_subplot(122)
        skill_counts = pd.Series(all_skills).value_counts()[:10]
        
        bars = ax2.barh(
            range(len(skill_counts)),
            skill_counts.values,
            color=plt.cm.Pastel1(np.linspace(0, 1, len(skill_counts))),
            alpha=0.7
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(
                width, bar.get_y() + bar.get_height()/2,
                f'{int(width)}',
                ha='left', va='center',
                fontweight='bold'
            )
        
        ax2.set_title('Top 10 Most Common Skills', pad=20, fontsize=12)
        ax2.set_yticks(range(len(skill_counts)))
        ax2.set_yticklabels(skill_counts.index)
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_comparison_tab(self):
        """Create enhanced candidate comparison tab"""
        comparison_tab = ttk.Frame(self.notebook, style='Dashboard.TFrame')
        self.notebook.add(comparison_tab, text='Candidate Comparison')
        
        content_frame = self.create_scrollable_container(comparison_tab)
        
        # Instructions
        ttk.Label(
            content_frame,
            text="Select candidates to compare (2-3 candidates)",
            font=('Helvetica', 12, 'bold'),
            foreground=self.colors['primary']
        ).pack(pady=20)
        
        # Create selection area with cards
        selection_frame = ttk.Frame(content_frame, style='Dashboard.TFrame')
        selection_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Initialize selection tracking
        self.selected_candidates = []
        self.candidate_vars = {}
        
        # Create grid of candidate cards
        sorted_candidates = sorted(
            self.results,
            key=lambda x: x['similarity_score'],
            reverse=True
        )
        
        for i, candidate in enumerate(sorted_candidates):
            card = ttk.Frame(selection_frame, style='Card.TFrame')
            card.grid(row=i//3, column=i%3, padx=10, pady=10, sticky='nsew')
            
            # Configure grid column weights
            selection_frame.grid_columnconfigure(0, weight=1)
            selection_frame.grid_columnconfigure(1, weight=1)
            selection_frame.grid_columnconfigure(2, weight=1)
            
            # Candidate info
            name = candidate['name'] or f'Candidate {i+1}'
            score = candidate['similarity_score']
            
            ttk.Label(
                card,
                text=name,
                font=('Helvetica', 11, 'bold')
            ).pack(pady=(10, 5))
            
            ttk.Label(
                card,
                text=f"Match Score: {score:.1f}%",
                foreground=self.colors['accent']
            ).pack()
            
            # Skills preview
            all_skills = [s for skills in candidate['skills'].values() for s in skills]
            top_skills = ', '.join(all_skills[:3]) + ('...' if len(all_skills) > 3 else '')
            
            ttk.Label(
                card,
                text=f"Top Skills: {top_skills}",
                wraplength=200
            ).pack(pady=5)
            
            # Checkbox for selection
            var = tk.BooleanVar()
            self.candidate_vars[id(candidate)] = (var, candidate)
            
            ttk.Checkbutton(
                card,
                text="Select for comparison",
                variable=var,
                command=lambda c=id(candidate): self.update_comparison(c)
            ).pack(pady=(0, 10))
        
        # Comparison visualization area
        self.comparison_frame = ttk.Frame(content_frame, style='Dashboard.TFrame')
        self.comparison_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
    def update_comparison(self, idx):
        """Handle candidate selection for comparison"""
        var, candidate = self.candidate_vars[idx]
        
        if var.get():
            if len(self.selected_candidates) >= 3:
                var.set(False)
                tk.messagebox.showwarning(
                    "Selection Limit",
                    "You can select maximum 3 candidates for comparison"
                )
                return
            self.selected_candidates.append(candidate)
        else:
            if candidate in self.selected_candidates:
                self.selected_candidates.remove(candidate)
        
        self.update_comparison_charts()
        
    def update_comparison_charts(self):
        """Update comparison visualization"""
        # Clear previous content
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()
        
        if len(self.selected_candidates) < 2:
            return
        
        # Create comparison visualizations
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.set_facecolor(self.colors['background'])
        
        # Radar chart for skills comparison
        ax1 = fig.add_subplot(121, projection='polar')
        
        # Prepare data for radar chart
        categories = ['Match Score', 'Skills', 'Experience', 'Education']
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        ax1.set_theta_offset(np.pi / 2)
        ax1.set_theta_direction(-1)
        ax1.set_rlabel_position(0)
        
        plt.xticks(angles[:-1], categories)
        
        for idx, candidate in enumerate(self.selected_candidates):
            values = [
                candidate['similarity_score'],
                len([s for skills in candidate['skills'].values() for s in skills]),
                len(candidate['job_titles']),
                len(candidate['education'])
            ]
            
            # Normalize values
            max_values = [100, 50, 10, 5]  # Approximate max values for each category
            values = [v/m*100 for v, m in zip(values, max_values)]
            values += values[:1]
            
            ax1.plot(angles, values, linewidth=1, linestyle='solid', label=candidate['name'])
            ax1.fill(angles, values, alpha=0.1)
        
        ax1.set_title("Candidate Comparison (Normalized)", pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Detailed comparison heatmap
        ax2 = fig.add_subplot(122)
        
        data = []
        labels = []
        
        for candidate in self.selected_candidates:
            skills_count = len([s for skills in candidate['skills'].values() for s in skills])
            data.append([
                candidate['similarity_score'],
                skills_count,
                len(candidate['job_titles']),
                len(candidate['education'])
            ])
            labels.append(candidate['name'] or 'Unknown')
        
        df = pd.DataFrame(
            data,
            columns=categories,
            index=labels
        )
        
        sns.heatmap(df, annot=True, cmap='YlOrRd', ax=ax2, fmt='.1f')
        ax2.set_title('Detailed Comparison', pad=20)
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_recommendations_tab(self):
        """Create recommendations tab"""
        recommendations_tab = ttk.Frame(self.notebook, style='Dashboard.TFrame')
        self.notebook.add(recommendations_tab, text='Recommendations')
        
        content_frame = self.create_scrollable_container(recommendations_tab)
        
        # Top candidates section
        top_label = ttk.Label(
            content_frame,
            text="Top Candidate Recommendations",
            font=('Helvetica', 14, 'bold'),
            foreground=self.colors['primary']
        )
        top_label.pack(pady=20)
        
        # Sort candidates by score
        sorted_candidates = sorted(
            self.results,
            key=lambda x: x['similarity_score'],
            reverse=True
        )
        
        # Create candidate cards container
        cards_frame = ttk.Frame(content_frame, style='Dashboard.TFrame')
        cards_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create candidate cards
        for i, candidate in enumerate(sorted_candidates[:10]):  # Show top 10
            self.create_candidate_card(cards_frame, candidate, i+1)
        
    def create_candidate_card(self, parent, candidate, rank):
        """Create a detailed candidate card"""
        card = ttk.Frame(parent, style='Card.TFrame')
        card.pack(fill=tk.X, pady=10)
        
        # Header with rank and score
        header_frame = ttk.Frame(card)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        rank_label = ttk.Label(
            header_frame,
            text=f"#{rank}",
            font=('Helvetica', 12, 'bold'),
            foreground=self.colors['accent']
        )
        rank_label.pack(side=tk.LEFT)
        
        score_label = ttk.Label(
            header_frame,
            text=f"Match Score: {candidate['similarity_score']:.1f}%",
            font=('Helvetica', 12),
            foreground=self.colors['primary']
        )
        score_label.pack(side=tk.RIGHT)
        
        # Candidate details
        details_frame = ttk.Frame(card)
        details_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left column - Basic info
        left_col = ttk.Frame(details_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(
            left_col,
            text=f"Name: {candidate['name'] or 'Not specified'}",
            font=('Helvetica', 10)
        ).pack(anchor='w')
        
        ttk.Label(
            left_col,
            text=f"Email: {candidate['email'] or 'Not specified'}",
            font=('Helvetica', 10)
        ).pack(anchor='w')
        
        ttk.Label(
            left_col,
            text=f"Phone: {candidate['phone'] or 'Not specified'}",
            font=('Helvetica', 10)
        ).pack(anchor='w')
        
        # Right column - Skills and education
        right_col = ttk.Frame(details_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Top skills
        all_skills = [skill for skills in candidate['skills'].values() for skill in skills]
        top_skills = ', '.join(all_skills[:5])
        ttk.Label(
            right_col,
            text=f"Top Skills: {top_skills}",
            font=('Helvetica', 10)
        ).pack(anchor='w')
        
        # Education
        education = candidate['education'][0] if candidate['education'] else 'Not specified'
        ttk.Label(
            right_col,
            text=f"Education: {education}",
            font=('Helvetica', 10)
        ).pack(anchor='w')
        
        # Action buttons
        button_frame = ttk.Frame(card)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        view_btn = ttk.Button(
            button_frame,
            text="View Details",
            command=lambda: self.show_candidate_details(candidate)
        )
        view_btn.pack(side=tk.LEFT, padx=5)
        
        email_btn = ttk.Button(
            button_frame,
            text="Send Email",
            command=lambda: self.prepare_email(candidate)
        )
        email_btn.pack(side=tk.LEFT, padx=5)
        
    def create_email_tab(self):
        """Create enhanced email management tab"""
        email_tab = ttk.Frame(self.notebook, style='Dashboard.TFrame')
        self.notebook.add(email_tab, text='Email Management')
        
        content_frame = self.create_scrollable_container(email_tab)
        
        # Email settings frame with better styling
        settings_frame = ttk.LabelFrame(
            content_frame,
            text="Email Configuration",
            padding="15",
            style='Card.TFrame'
        )
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # SMTP settings with grid layout
        grid_frame = ttk.Frame(settings_frame)
        grid_frame.pack(fill=tk.X, pady=5)
        
        # Configure grid columns
        for i in range(4):
            grid_frame.columnconfigure(i, weight=1)
        
        # SMTP Server
        ttk.Label(
            grid_frame,
            text="SMTP Server:",
            font=('Helvetica', 10, 'bold')
        ).grid(row=0, column=0, padx=5, pady=5, sticky='e')
        
        self.smtp_entry = ttk.Entry(grid_frame)
        self.smtp_entry.insert(0, "smtp.gmail.com")
        self.smtp_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Port
        ttk.Label(
            grid_frame,
            text="Port:",
            font=('Helvetica', 10, 'bold')
        ).grid(row=0, column=2, padx=5, pady=5, sticky='e')
        
        self.port_entry = ttk.Entry(grid_frame, width=10)
        self.port_entry.insert(0, "587")
        self.port_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        
        # Email
        ttk.Label(
            grid_frame,
            text="Email:",
            font=('Helvetica', 10, 'bold')
        ).grid(row=1, column=0, padx=5, pady=5, sticky='e')
        
        self.email_entry = ttk.Entry(grid_frame)
        self.email_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Password
        ttk.Label(
            grid_frame,
            text="Password:",
            font=('Helvetica', 10, 'bold')
        ).grid(row=1, column=2, padx=5, pady=5, sticky='e')
        
        self.password_entry = ttk.Entry(grid_frame, show="*")
        self.password_entry.grid(row=1, column=3, padx=5, pady=5, sticky='ew')
        
        # Test connection button
        test_btn = ttk.Button(
            settings_frame,
            text="Test Connection",
            command=self.test_email_connection
        )
        test_btn.pack(pady=10)
        
        # Email templates section
        templates_frame = ttk.LabelFrame(
            content_frame,
            text="Email Templates",
            padding="15",
            style='Card.TFrame'
        )
        templates_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Template selection
        template_select_frame = ttk.Frame(templates_frame)
        template_select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            template_select_frame,
            text="Select Template:",
            font=('Helvetica', 10, 'bold')
        ).pack(side=tk.LEFT, padx=5)
        
        templates = ["Interview Invitation", "Rejection", "Follow-up", "Custom"]
        self.template_var = tk.StringVar(value=templates[0])
        template_combo = ttk.Combobox(
            template_select_frame,
            values=templates,
            textvariable=self.template_var,
            state='readonly'
        )
        template_combo.pack(side=tk.LEFT, padx=5)
        template_combo.bind('<<ComboboxSelected>>', self.load_template)
        
        # Template editor
        editor_frame = ttk.Frame(templates_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Subject line
        subject_frame = ttk.Frame(editor_frame)
        subject_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            subject_frame,
            text="Subject:",
            font=('Helvetica', 10, 'bold')
        ).pack(side=tk.LEFT, padx=5)
        
        self.subject_entry = ttk.Entry(subject_frame)
        self.subject_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Template content
        self.template_text = scrolledtext.ScrolledText(
            editor_frame,
            height=15,
            font=('Helvetica', 10)
        )
        self.template_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Available variables info
        variables_frame = ttk.LabelFrame(
            editor_frame,
            text="Available Variables",
            padding="5"
        )
        variables_frame.pack(fill=tk.X, pady=5)
        
        variables_text = """
        {name} - Candidate's name
        {score} - Match score
        {email} - Candidate's email
        {phone} - Candidate's phone
        {skills} - Top skills
        """
        ttk.Label(
            variables_frame,
            text=variables_text,
            justify=tk.LEFT,
            font=('Helvetica', 9)
        ).pack(padx=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(templates_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            buttons_frame,
            text="Save Template",
            command=self.save_template
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Reset",
            command=self.reset_template
        ).pack(side=tk.LEFT, padx=5)

    def test_email_connection(self):
        """Test SMTP connection"""
        try:
            server = smtplib.SMTP(self.smtp_entry.get(), int(self.port_entry.get()))
            server.starttls()
            server.login(self.email_entry.get(), self.password_entry.get())
            server.quit()
            messagebox.showinfo("Success", "Connection test successful!")
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")

    def load_template(self, event=None):
        """Load selected email template"""
        templates = {
            "Interview Invitation": {
                "subject": "Interview Invitation - {score}% Match",
                "content": """Dear {name},

Based on our initial screening, your profile shows a strong {score}% match with our requirements.

We would like to invite you for an interview to discuss your application further.

Best regards,
Recruitment Team"""
            },
            "Rejection": {
                "subject": "Application Status Update",
                "content": """Dear {name},

Thank you for your interest in our position. While your profile shows a {score}% match,
we have decided to proceed with other candidates whose qualifications better match our current needs.

We will keep your application on file for future opportunities.

Best regards,
Recruitment Team"""
            },
            "Follow-up": {
                "subject": "Application Follow-up",
                "content": """Dear {name},

We are following up on your application. Your profile shows a promising {score}% match with our requirements.

Could you please provide additional information about your experience with:
{skills}

Best regards,
Recruitment Team"""
            }
        }
        
        template_name = self.template_var.get()
        if template_name in templates:
            self.subject_entry.delete(0, tk.END)
            self.subject_entry.insert(0, templates[template_name]["subject"])
            
            self.template_text.delete('1.0', tk.END)
            self.template_text.insert('1.0', templates[template_name]["content"])

    def save_template(self):
        """Save current template"""
        template_name = self.template_var.get()
        if template_name != "Custom":
            if messagebox.askyesno("Save Template", 
                                  "Do you want to save changes to this template?"):
                messagebox.showinfo("Success", "Template saved successfully!")

    def reset_template(self):
        """Reset template to default"""
        if messagebox.askyesno("Reset Template", 
                              "Do you want to reset this template to default?"):
            self.load_template()
        
    def prepare_email(self, candidate):
        """Prepare email to candidate"""
        self.notebook.select(4)  # Switch to email tab
        
        # Fill template with candidate info
        template = self.template_text.get('1.0', tk.END)
        email_content = template.format(
            name=candidate['name'] or "Candidate",
            score=f"{candidate['similarity_score']:.1f}"
        )
        
        # Create email dialog
        dialog = tk.Toplevel(self.window)
        dialog.title("Send Email")
        dialog.geometry("600x400")
        
        ttk.Label(dialog, text="To:").pack(padx=20, pady=5, anchor='w')
        to_entry = ttk.Entry(dialog, width=50)
        to_entry.insert(0, candidate['email'])
        to_entry.pack(padx=20, pady=5)
        
        ttk.Label(dialog, text="Subject:").pack(padx=20, pady=5, anchor='w')
        subject_entry = ttk.Entry(dialog, width=50)
        subject_entry.insert(0, "Your Job Application")
        subject_entry.pack(padx=20, pady=5)
        
        ttk.Label(dialog, text="Message:").pack(padx=20, pady=5, anchor='w')
        message_text = scrolledtext.ScrolledText(dialog, height=10)
        message_text.insert('1.0', email_content)
        message_text.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)
        
        send_btn = ttk.Button(
            dialog,
            text="Send Email",
            command=lambda: self.send_email(
                to_entry.get(),
                subject_entry.get(),
                message_text.get('1.0', tk.END)
            )
        )
        send_btn.pack(pady=20)
        
    def send_email(self, to_email, subject, message):
        """Send email to candidate"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_entry.get()
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Create server connection
            server = smtplib.SMTP(self.smtp_entry.get(), int(self.port_entry.get()))
            server.starttls()
            server.login(self.email_entry.get(), self.password_entry.get())
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            messagebox.showinfo("Success", "Email sent successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send email: {str(e)}")
            
    def show_candidate_details(self, candidate):
        """Show detailed candidate information"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Candidate Details")
        dialog.geometry("800x600")
        
        # Create scrolled text widget
        text = scrolledtext.ScrolledText(
            dialog,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=('Helvetica', 10)
        )
        text.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Configure tags for formatting
        text.tag_configure('heading', font=('Helvetica', 12, 'bold'))
        text.tag_configure('subheading', font=('Helvetica', 10, 'bold'))
        
        # Insert candidate details
        text.insert(tk.END, "Candidate Details\n\n", 'heading')
        
        details = f"""
Name: {candidate['name'] or 'Not specified'}
Email: {candidate['email'] or 'Not specified'}
Phone: {candidate['phone'] or 'Not specified'}
Match Score: {candidate['similarity_score']:.1f}%\n
"""
        text.insert(tk.END, details)
        
        # Education
        text.insert(tk.END, "\nEducation:\n", 'subheading')
        for edu in candidate['education']:
            text.insert(tk.END, f"• {edu}\n")
            
        # Experience
        text.insert(tk.END, "\nJob Titles:\n", 'subheading')
        for title in candidate['job_titles']:
            text.insert(tk.END, f"• {title}\n")
            
        # Skills by category
        text.insert(tk.END, "\nSkills:\n", 'subheading')
        for category, skills in candidate['skills'].items():
            text.insert(tk.END, f"\n{category}:\n")
            for skill in skills:
                text.insert(tk.END, f"• {skill}\n")
                
        text.configure(state='disabled') 