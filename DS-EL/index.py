import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import os
import threading
import io
from contextlib import redirect_stdout

class DataScienceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Science EDA & ML Training Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_column = None
        self.model = None
        self.feature_names = None
        self.encoded_classes = None
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.LabelFrame(main_frame, text="Data Input & Controls")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Select Data File:").pack(side=tk.LEFT, padx=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        
        cleaning_frame = ttk.LabelFrame(left_frame, text="Data Cleaning Options")
        cleaning_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.handle_missing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cleaning_frame, text="Handle Missing Values", variable=self.handle_missing_var).pack(anchor=tk.W, padx=5, pady=2)
        
        missing_frame = ttk.Frame(cleaning_frame)
        missing_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(missing_frame, text="Strategy:").pack(side=tk.LEFT, padx=5)
        self.missing_strategy = tk.StringVar(value="mean")
        strategy_combo = ttk.Combobox(missing_frame, textvariable=self.missing_strategy, width=15)
        strategy_combo['values'] = ['mean', 'median', 'most_frequent', 'constant']
        strategy_combo.pack(side=tk.LEFT, padx=5)
        
        self.remove_outliers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cleaning_frame, text="Remove Outliers (IQR method)", variable=self.remove_outliers_var).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Button(left_frame, text="Analyze Data", command=self.run_eda).pack(fill=tk.X, padx=5, pady=10)
        
        target_frame = ttk.Frame(left_frame)
        target_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(target_frame, text="Target Column:").pack(side=tk.LEFT, padx=5)
        self.target_var = tk.StringVar()
        self.target_dropdown = ttk.Combobox(target_frame, textvariable=self.target_var, state="disabled")
        self.target_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.feature_selection_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Use Feature Selection", variable=self.feature_selection_var).pack(anchor=tk.W, padx=5, pady=2)
        
        model_frame = ttk.LabelFrame(left_frame, text="Model Selection")
        model_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.model_var = tk.StringVar(value="knn")
        ttk.Radiobutton(model_frame, text="K-Nearest Neighbors", variable=self.model_var, value="knn").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(model_frame, text="Decision Tree", variable=self.model_var, value="dt").pack(anchor=tk.W, padx=5, pady=2)
        
        self.hyper_tuning_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(model_frame, text="Use Hyperparameter Tuning", variable=self.hyper_tuning_var).pack(anchor=tk.W, padx=5, pady=2)
        
        params_frame = ttk.LabelFrame(left_frame, text="Model Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=10)
        
        knn_frame = ttk.Frame(params_frame)
        knn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(knn_frame, text="KNN Neighbors:").pack(side=tk.LEFT, padx=5)
        self.knn_n = tk.IntVar(value=5)
        ttk.Spinbox(knn_frame, from_=1, to=20, textvariable=self.knn_n, width=5).pack(side=tk.LEFT, padx=5)
        
        dt_frame = ttk.Frame(params_frame)
        dt_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(dt_frame, text="Max Depth:").pack(side=tk.LEFT, padx=5)
        self.dt_depth = tk.IntVar(value=5)
        ttk.Spinbox(dt_frame, from_=1, to=20, textvariable=self.dt_depth, width=5).pack(side=tk.LEFT, padx=5)
        
        split_frame = ttk.Frame(params_frame)
        split_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(split_frame, text="Test Split:").pack(side=tk.LEFT, padx=5)
        self.test_split = tk.DoubleVar(value=0.2)
        ttk.Spinbox(split_frame, from_=0.1, to=0.5, increment=0.05, textvariable=self.test_split, width=5).pack(side=tk.LEFT, padx=5)
        
        cv_frame = ttk.Frame(params_frame)
        cv_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(cv_frame, text="Cross-validation Folds:").pack(side=tk.LEFT, padx=5)
        self.cv_folds = tk.IntVar(value=5)
        ttk.Spinbox(cv_frame, from_=2, to=10, textvariable=self.cv_folds, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_frame, text="Train Model", command=self.train_model).pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(left_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, padx=5, pady=10)
        
        right_frame = ttk.Notebook(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.eda_tab = ttk.Frame(right_frame)
        right_frame.add(self.eda_tab, text="EDA Results")
        
        self.eda_output = scrolledtext.ScrolledText(self.eda_tab, wrap=tk.WORD)
        self.eda_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.viz_tab = ttk.Frame(right_frame)
        right_frame.add(self.viz_tab, text="Visualizations")
        
        self.model_tab = ttk.Frame(right_frame)
        right_frame.add(self.model_tab, text="Model Results")
        
        self.model_output = scrolledtext.ScrolledText(self.model_tab, wrap=tk.WORD)
        self.model_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.model_viz_tab = ttk.Frame(right_frame)
        right_frame.add(self.model_viz_tab, text="Model Visualization")
        
        self.pred_tab = ttk.Frame(right_frame)
        right_frame.add(self.pred_tab, text="Make Predictions")
        
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")
    
    def browse_file(self):
        filetypes = [
            ("CSV files", "*.csv"), 
            ("Excel files", "*.xlsx"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.file_path_var.set(filename)
            self.status_var.set(f"File selected: {os.path.basename(filename)}")
    
    def load_data(self):
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file first")
            return False
        
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format. Please use CSV or Excel files.")
                return False
            
            self.original_df = self.df.copy() 
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            return False
    
    def clean_data(self):
        """Apply data cleaning based on user selections"""
        if self.handle_missing_var.get():
            num_cols = self.df.select_dtypes(include=np.number).columns
            if not num_cols.empty:
                imputer = SimpleImputer(strategy=self.missing_strategy.get())
                self.df[num_cols] = imputer.fit_transform(self.df[num_cols])
            
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])
        
        if self.remove_outliers_var.get():
            num_cols = self.df.select_dtypes(include=np.number).columns
            original_rows = len(self.df)
            
            for col in num_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
            removed_rows = original_rows - len(self.df)
            if removed_rows > 0:
                self.eda_output.insert(tk.END, f"\nOutlier Removal\n{'='*50}\n")
                self.eda_output.insert(tk.END, f"Removed {removed_rows} rows ({removed_rows/original_rows*100:.2f}% of data) as outliers\n")
    
    def run_eda(self):
        if not self.load_data():
            return
        
        self.status_var.set("Running EDA analysis...")
        
        threading.Thread(target=self._run_eda_thread, daemon=True).start()
    
    def _run_eda_thread(self):
        self.eda_output.delete(1.0, tk.END)
        for widget in self.viz_tab.winfo_children():
            widget.destroy()
        
        self.eda_output.insert(tk.END, f"Data Cleaning\n{'='*50}\n")
        if self.handle_missing_var.get():
            missing_before = self.df.isnull().sum().sum()
            self.eda_output.insert(tk.END, f"Missing values (before cleaning): {missing_before}\n")
            self.eda_output.insert(tk.END, f"Missing values strategy: {self.missing_strategy.get()}\n")
        
        if self.remove_outliers_var.get():
            self.eda_output.insert(tk.END, "Outlier removal: Enabled (using IQR method)\n")
        
        self.clean_data()
        
        self.root.after(0, lambda: self._update_target_dropdown())
        
        self.eda_output.insert(tk.END, f"\nDataset Overview\n{'='*50}\n")
        self.eda_output.insert(tk.END, f"Number of rows: {self.df.shape[0]}\n")
        self.eda_output.insert(tk.END, f"Number of columns: {self.df.shape[1]}\n\n")
        
        self.eda_output.insert(tk.END, f"Sample Data (first 5 rows)\n{'-'*50}\n")
        sample_data = self.df.head().to_string()
        self.eda_output.insert(tk.END, f"{sample_data}\n\n")
        
        self.eda_output.insert(tk.END, f"Data Types\n{'='*50}\n")
        dtypes_summary = self.df.dtypes.value_counts().to_dict()
        for dtype, count in dtypes_summary.items():
            self.eda_output.insert(tk.END, f"{dtype}: {count} columns\n")
        
        self.eda_output.insert(tk.END, f"\nColumn Details\n{'-'*50}\n")
        for col, dtype in self.df.dtypes.items():
            self.eda_output.insert(tk.END, f"{col}: {dtype}\n")
        
        self.eda_output.insert(tk.END, f"\nMissing Values\n{'='*50}\n")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            self.eda_output.insert(tk.END, "No missing values found\n")
        else:
            for col, count in missing.items():
                self.eda_output.insert(tk.END, f"{col}: {count} missing values ({(count/len(self.df)*100):.2f}%)\n")
        
        num_cols = self.df.select_dtypes(include=np.number).columns
        if not num_cols.empty:
            self.eda_output.insert(tk.END, f"\nNumerical Columns Summary\n{'='*50}\n")
            stats = self.df[num_cols].describe().transpose()
            stats['range'] = stats['max'] - stats['min']
            stats['IQR'] = stats['75%'] - stats['25%']
            
            stats['skewness'] = self.df[num_cols].skew()
            
            for col in stats.index:
                self.eda_output.insert(tk.END, f"\n{col}:\n")
                self.eda_output.insert(tk.END, f"  Min: {stats.loc[col, 'min']:.2f}, Max: {stats.loc[col, 'max']:.2f}, Range: {stats.loc[col, 'range']:.2f}\n")
                self.eda_output.insert(tk.END, f"  Mean: {stats.loc[col, 'mean']:.2f}, Median: {stats.loc[col, '50%']:.2f}\n")
                self.eda_output.insert(tk.END, f"  Std Dev: {stats.loc[col, 'std']:.2f}, IQR: {stats.loc[col, 'IQR']:.2f}\n")
                self.eda_output.insert(tk.END, f"  Skewness: {stats.loc[col, 'skewness']:.2f}\n")
        
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if not cat_cols.empty:
            self.eda_output.insert(tk.END, f"\nCategorical Columns Summary\n{'='*50}\n")
            for col in cat_cols:
                value_counts = self.df[col].value_counts()
                unique_count = len(value_counts)
                
                self.eda_output.insert(tk.END, f"\n{col}:\n")
                self.eda_output.insert(tk.END, f"  Unique values: {unique_count}\n")
                
                if unique_count <= 10:
                    for val, count in value_counts.items():
                        self.eda_output.insert(tk.END, f"  {val}: {count} ({(count/len(self.df)*100):.2f}%)\n")
                else:
                    top5 = value_counts.head(5)
                    for val, count in top5.items():
                        self.eda_output.insert(tk.END, f"  {val}: {count} ({(count/len(self.df)*100):.2f}%)\n")
                    self.eda_output.insert(tk.END, f"  (and {unique_count-5} more values)\n")
        
        self.eda_output.insert(tk.END, f"\nKey Insights\n{'='*50}\n")
        
        if len(num_cols) > 1:
            corr_matrix = self.df[num_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [(col1, col2, corr_matrix.loc[col1, col2]) 
                         for col1 in upper_tri.index 
                         for col2 in upper_tri.columns 
                         if upper_tri.loc[col1, col2] > 0.7]
            
            if high_corr:
                self.eda_output.insert(tk.END, "Highly correlated features:\n")
                for col1, col2, corr_val in high_corr:
                    self.eda_output.insert(tk.END, f"  {col1} and {col2}: {corr_val:.2f}\n")
            else:
                self.eda_output.insert(tk.END, "No highly correlated numerical features found\n")
        
        highly_skewed = [(col, self.df[col].skew()) for col in num_cols if abs(self.df[col].skew()) > 1]
        if highly_skewed:
            self.eda_output.insert(tk.END, "\nHighly skewed features (might need transformation):\n")
            for col, skew_val in highly_skewed:
                self.eda_output.insert(tk.END, f"  {col}: {skew_val:.2f}\n")
        
        for col in cat_cols:
            value_counts = self.df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.8:  # If dominant class > 80%
                self.eda_output.insert(tk.END, f"\nImbalanced categorical column: {col}\n")
                self.eda_output.insert(tk.END, f"  Dominant class: {value_counts.index[0]} ({value_counts.iloc[0]*100:.2f}%)\n")
        
        self.root.after(0, lambda: self._create_visualizations())
        
        self.status_var.set("EDA analysis completed")
    
    def _update_target_dropdown(self):
        self.target_dropdown['values'] = list(self.df.columns)
        self.target_dropdown['state'] = 'readonly'
        
        self._create_prediction_tab()
    
    def _create_prediction_tab(self):
        for widget in self.pred_tab.winfo_children():
            widget.destroy()
            
        input_frame = ttk.LabelFrame(self.pred_tab, text="Input Values for Prediction")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Train a model first to enable predictions").pack(pady=20)
    
    def _create_visualizations(self):
        viz_notebook = ttk.Notebook(self.viz_tab)
        viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        num_df = self.df.select_dtypes(include=np.number)
        if num_df.shape[1] > 1:  
            corr_tab = ttk.Frame(viz_notebook)
            viz_notebook.add(corr_tab, text="Correlation Matrix")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = num_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f", linewidths=0.5)
            ax.set_title("Correlation Matrix")
            
            canvas = FigureCanvasTkAgg(fig, master=corr_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        num_cols = self.df.select_dtypes(include=np.number).columns
        if not num_cols.empty:
            dist_tab = ttk.Frame(viz_notebook)
            viz_notebook.add(dist_tab, text="Distributions")
            
            canvas_frame = ttk.Frame(dist_tab)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            canvas = tk.Canvas(canvas_frame)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            for i, col in enumerate(num_cols[:min(10, len(num_cols))]):  
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.histplot(self.df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                
                plot_frame = ttk.Frame(scrollable_frame)
                plot_frame.pack(pady=10)
                
                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()
        
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if not cat_cols.empty:
            cat_tab = ttk.Frame(viz_notebook)
            viz_notebook.add(cat_tab, text="Categorical Data")
            
            canvas_frame = ttk.Frame(cat_tab)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            canvas = tk.Canvas(canvas_frame)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            for i, col in enumerate(cat_cols[:min(5, len(cat_cols))]):  
                value_counts = self.df[col].value_counts().head(10)  
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                ax.set_title(f"Top Categories in {col}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                plot_frame = ttk.Frame(scrollable_frame)
                plot_frame.pack(pady=10)
                
                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()
        
        if len(num_cols) >= 2:
            scatter_tab = ttk.Frame(viz_notebook)
            viz_notebook.add(scatter_tab, text="Scatter Matrix")
            
            plot_cols = num_cols[:min(5, len(num_cols))]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.pairplot(self.df[plot_cols])
            
            canvas = FigureCanvasTkAgg(plt.gcf(), master=scatter_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load and analyze data first")
            return
        
        target = self.target_var.get()
        if not target:
            messagebox.showerror("Error", "Please select a target column")
            return
        
        self.status_var.set("Training model...")
        
        threading.Thread(target=lambda: self._train_model_thread(target), daemon=True).start()
    
    def _train_model_thread(self, target):
        self.model_output.delete(1.0, tk.END)
        for widget in self.model_viz_tab.winfo_children():
            widget.destroy()
        
        try:
            if self.df[target].dtype == 'object' or self.df[target].nunique() < 10:
                model_type = "Classification"
                if self.df[target].dtype == 'object':
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(self.df[target])
                    self.encoded_classes = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
                else:
                    y = self.df[target]
                    self.encoded_classes = None
            else:
                messagebox.showerror("Error", "This app currently only supports classification tasks. The target column has too many unique values.")
                return
            
            X = self.df.drop(target, axis=1)
            
            self.target_column = target
            
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
            self.feature_names = X.columns.tolist()
            
            test_size = self.test_split.get()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                print(f"Model Training: {model_type}\n{'='*50}")
                print(f"Target Column: {target}")
                print(f"Number of features: {X.shape[1]}")
                print(f"Training samples: {len(self.X_train)}")
                print(f"Testing samples: {len(self.X_test)}")
                
                model_name = self.model_var.get()
                
                if model_name == "knn":
                    if self.hyper_tuning_var.get():
                        print("\nPerforming hyperparameter tuning for KNN...")
                        param_grid = {
                            'n_neighbors': range(1, 21, 2),
                            'weights': ['uniform', 'distance'],
                            'p': [1, 2]
                        }
                        
                        base_model = KNeighborsClassifier()
                        grid_search = GridSearchCV(
                            base_model, param_grid, cv=self.cv_folds.get(), scoring='accuracy')
                        grid_search.fit(self.X_train, self.y_train)
                        
                        self.model = grid_search.best_estimator_
                        print(f"Best parameters: {grid_search.best_params_}")
                    else:
                        n_neighbors = self.knn_n.get()
                        print(f"\nTraining KNN model with n_neighbors={n_neighbors}")
                        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
                        self.model.fit(self.X_train, self.y_train)
                
                elif model_name == "dt":
                    if self.hyper_tuning_var.get():
                        print("\nPerforming hyperparameter tuning for Decision Tree...")
                        param_grid = {
                            'max_depth': range(1, 21),
                            'min_samples_split': [2, 5, 10],
                            'criterion': ['gini', 'entropy']
                        }
                        
                        base_model = DecisionTreeClassifier(random_state=42)
                        grid_search = GridSearchCV(
                            base_model, param_grid, cv=self.cv_folds.get(), scoring='accuracy')
                        grid_search.fit(self.X_train, self.y_train)
                        
                        self.model = grid_search.best_estimator_
                        print(f"Best parameters: {grid_search.best_params_}")
                    else:
                        max_depth = self.dt_depth.get()
                        print(f"\nTraining Decision Tree model with max_depth={max_depth}")
                        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                        self.model.fit(self.X_train, self.y_train)
                
                train_preds = self.model.predict(self.X_train)
                test_preds = self.model.predict(self.X_test)
                
                print("\nModel Performance\n{'-'*50}")
                print(f"Training accuracy: {accuracy_score(self.y_train, train_preds):.4f}")
                print(f"Testing accuracy: {accuracy_score(self.y_test, test_preds):.4f}")
                
                print("\nClassification Report\n{'-'*50}")
                print(classification_report(self.y_test, test_preds))
                
                print("\nConfusion Matrix\n{'-'*50}")
                cm = confusion_matrix(self.y_test, test_preds)
                print(cm)
            
            self.model_output.delete(1.0, tk.END)
            self.model_output.insert(tk.END, output_buffer.getvalue())
            
            self.root.after(0, lambda: self._visualize_model())
            
            self.root.after(0, lambda: self._update_prediction_tab())
            
            self.status_var.set("Model trained successfully")
        
        except Exception as e:
            self.model_output.delete(1.0, tk.END)
            self.model_output.insert(tk.END, f"Error during model training: {str(e)}")
            self.status_var.set("Error during model training")
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def _visualize_model(self):
        for widget in self.model_viz_tab.winfo_children():
            widget.destroy()
        
        viz_frame = ttk.Frame(self.model_viz_tab)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        model_name = self.model_var.get()
        
        if model_name == "dt":
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(self.model, filled=True, feature_names=self.feature_names, ax=ax, 
                      class_names=[str(c) for c in sorted(self.encoded_classes.keys())] if self.encoded_classes else None)
            ax.set_title("Decision Tree Visualization")
            
            canvas = FigureCanvasTkAgg(fig, master=viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        
        cm_frame = ttk.LabelFrame(viz_frame, text="Confusion Matrix")
        cm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        try:
            if hasattr(self.model, "predict_proba"):
                fig, ax = plt.subplots(figsize=(8, 6))
                y_prob = self.model.predict_proba(self.X_test)
                
                if y_prob.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(self.y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--', lw=2)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic')
                    ax.legend(loc="lower right")
                    
                    roc_frame = ttk.LabelFrame(viz_frame, text="ROC Curve")
                    roc_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    canvas = FigureCanvasTkAgg(fig, master=roc_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception:
            pass
    
    def _update_prediction_tab(self):
        for widget in self.pred_tab.winfo_children():
            widget.destroy()
            
        input_frame = ttk.LabelFrame(self.pred_tab, text="Input Values for Prediction")
        input_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        self.prediction_entries = {}
        
        feature_names = [col for col in self.df.columns if col != self.target_column]
        
        canvas = tk.Canvas(input_frame)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for i, feature in enumerate(feature_names):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=f"{feature}:").pack(side=tk.LEFT, padx=5)
            entry = ttk.Entry(frame)
            entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            if self.df[feature].dtype == 'object':
                values = self.df[feature].unique().tolist()
                combo = ttk.Combobox(frame, values=values)
                combo.pack(side=tk.RIGHT, padx=5)
                self.prediction_entries[feature] = combo
            else:
                try:
                    placeholder = f"e.g. {self.df[feature].mean():.2f}"
                    entry.insert(0, placeholder)
                except:
                    pass
                self.prediction_entries[feature] = entry
        
        output_frame = ttk.LabelFrame(self.pred_tab, text="Prediction Result")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.prediction_output = scrolledtext.ScrolledText(output_frame, height=10)
        self.prediction_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        predict_btn = ttk.Button(self.pred_tab, text="Make Prediction", command=self._predict)
        predict_btn.pack(pady=10)
    
    def _predict(self):
        try:
            input_data = {}
            for feature, entry in self.prediction_entries.items():
                try:
                    value = entry.get()
                    if value.startswith('e.g.'):
                        value = str(self.df[feature].mean())
                    input_data[feature] = value
                except:
                    messagebox.showerror("Error", f"Invalid input for {feature}")
                    return
            
            input_df = pd.DataFrame([input_data])
            
            cat_cols = input_df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
            
            for col in self.feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[self.feature_names]
            
            prediction = self.model.predict(input_df)[0]
            
            if self.encoded_classes and prediction in self.encoded_classes:
                prediction_label = self.encoded_classes[prediction]
            else:
                prediction_label = prediction
            
            self.prediction_output.delete(1.0, tk.END)
            self.prediction_output.insert(tk.END, f"Predicted {self.target_column}: {prediction_label}\n\n")
            
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(input_df)[0]
                self.prediction_output.insert(tk.END, "Prediction Probabilities:\n")
                for i, p in enumerate(proba):
                    class_label = self.encoded_classes[i] if self.encoded_classes else i
                    self.prediction_output.insert(tk.END, f"Class {class_label}: {p:.4f}\n")
            
        except Exception as e:
            self.prediction_output.delete(1.0, tk.END)
            self.prediction_output.insert(tk.END, f"Error during prediction: {str(e)}")
            
    def export_results(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train a model first")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                f.write("DATA SCIENCE APP - RESULTS EXPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("EDA RESULTS\n")
                f.write("-" * 50 + "\n")
                f.write(self.eda_output.get(1.0, tk.END))
                
                f.write("\n\nMODEL RESULTS\n")
                f.write("-" * 50 + "\n")
                f.write(self.model_output.get(1.0, tk.END))
                
                f.write("\n\nExported on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataScienceApp(root)
    root.mainloop()