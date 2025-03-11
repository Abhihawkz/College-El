import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time

# Set page configuration
st.set_page_config(
    page_title="Data Science Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
    }
    .css-1aumxhk {
        background-color: #e0f7fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1e88e5;
    }
    .metric-card {
        background-color: #fff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'preprocessed' not in st.session_state:
    st.session_state['preprocessed'] = False
if 'knn_model' not in st.session_state:
    st.session_state['knn_model'] = None
if 'dt_model' not in st.session_state:
    st.session_state['dt_model'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'target_name' not in st.session_state:
    st.session_state['target_name'] = None
if 'categorical_columns' not in st.session_state:
    st.session_state['categorical_columns'] = []
if 'numerical_columns' not in st.session_state:
    st.session_state['numerical_columns'] = []
if 'encoders' not in st.session_state:
    st.session_state['encoders'] = {}
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = False

# Create sidebar navigation
with st.sidebar:
    st.image("https://static.vecteezy.com/system/resources/previews/008/147/389/non_2x/technology-science-logo-with-data-concept-vector.jpg", width=50)
    st.title("Data Science Explorer")
    
    selected = option_menu(
    "Navigation",
    ["Data Upload", "Data Preprocessing", "Data Visualization", "Model Training", "Model Evaluation", "Prediction"],
    icons=["upload", "gear", "bar-chart", "cpu", "graph-up", "magic"],
    menu_icon="cast",
    default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee", "color": "black"},
        "nav-link-selected": {"background-color": "#02ab21", "color": "black"},
        "menu-title": {"color": "black"}  # Ensures "Navigation" text is black
    }
)


# Function to create a metrics card
def metric_card(title, value, description=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div>{description}</div>
    </div>
    """, unsafe_allow_html=True)

# 1. DATA UPLOAD PAGE
if selected == "Data Upload":
    st.title("📂 Data Upload")
    st.write("Upload your dataset to get started with the analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        with st.spinner('Loading data...'):
            data = pd.read_csv(uploaded_file)
            st.session_state['data'] = data
            
        st.success("Data successfully loaded!")
        
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            metric_card("Rows", f"{data.shape[0]:,}")
            metric_card("Missing Values", f"{data.isna().sum().sum():,}")
        with col2:
            metric_card("Columns", f"{data.shape[1]:,}")
            metric_card("Duplicates", f"{data.duplicated().sum():,}")
        
        # Basic dataset statistics
        st.subheader("Dataset Statistics")
        st.dataframe(data.describe())
        
        # Data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame(data.dtypes, columns=['Data Type'])
        dtype_df = dtype_df.reset_index().rename(columns={'index': 'Column'})
        st.dataframe(dtype_df)

# 2. DATA PREPROCESSING PAGE
elif selected == "Data Preprocessing":
    st.title("⚙️ Data Preprocessing")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first!")
    else:
        data = st.session_state['data']
        
        st.subheader("Select Target Variable")
        target_name = st.selectbox("Choose the target column", data.columns.tolist())
        st.session_state['target_name'] = target_name
        
        st.subheader("Handle Missing Values")
        missing_cols = data.columns[data.isna().any()].tolist()
        
        if missing_cols:
            st.write("Columns with missing values:", ", ".join(missing_cols))
            missing_strategy = st.radio(
                "Choose a strategy for handling missing values:",
                ["Drop rows with missing values", "Fill with mean/mode"]
            )
        else:
            st.write("No missing values found in the dataset.")
            missing_strategy = None
            
        st.subheader("Feature Selection")
        feature_cols = st.multiselect(
            "Select features for model training",
            [col for col in data.columns if col != target_name],
            default=[col for col in data.columns if col != target_name]
        )
        st.session_state['feature_names'] = feature_cols
        
        if feature_cols:
            # Identify categorical and numerical columns
            categorical_cols = []
            numerical_cols = []
            for col in feature_cols:
                if data[col].dtype == 'object' or data[col].nunique() < 10:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            
            st.session_state['categorical_columns'] = categorical_cols
            st.session_state['numerical_columns'] = numerical_cols
            
            if categorical_cols:
                st.subheader("Categorical Features")
                st.write("The following features will be encoded:", ", ".join(categorical_cols))
            
            if numerical_cols:
                st.subheader("Numerical Features")
                st.write("The following features will be scaled:", ", ".join(numerical_cols))
            
            # Train-test split options
            st.subheader("Train-Test Split")
            test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random state", 0, 100, 42)
            
            if st.button("Preprocess Data", key="preprocess_button"):
                with st.spinner("Preprocessing data..."):
                    # Make a copy of the data
                    processed_data = data.copy()
                    
                    # Handle missing values
                    if missing_strategy == "Drop rows with missing values":
                        processed_data = processed_data.dropna()
                    elif missing_strategy == "Fill with mean/mode":
                        for col in processed_data.columns:
                            if processed_data[col].dtype in ['int64', 'float64']:
                                processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                            else:
                                processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
                    
                    # Prepare features and target
                    X = processed_data[feature_cols]
                    y = processed_data[target_name]
                    
                    # Encode categorical features
                    encoders = {}
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        encoders[col] = le
                    st.session_state['encoders'] = encoders
                    
                    # Scale numerical features
                    if numerical_cols:
                        scaler = StandardScaler()
                        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                        st.session_state['scaler'] = scaler
                    
                    # Encode target if it's categorical
                    if y.dtype == 'object' or y.nunique() < 10:
                        target_encoder = LabelEncoder()
                        y = target_encoder.fit_transform(y.astype(str))
                        encoders['target'] = target_encoder
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Store in session state
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['preprocessed'] = True
                    
                    time.sleep(1)  # Simulate processing time
                
                st.success("Data preprocessing completed successfully!")
                
                # Show sample of processed data
                st.subheader("Processed Data Sample")
                processed_df = X.copy()
                processed_df[target_name] = y
                st.dataframe(processed_df.head())

# 3. DATA VISUALIZATION PAGE
elif selected == "Data Visualization":
    st.title("📊 Data Visualization")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first!")
    else:
        data = st.session_state['data']
        
        st.subheader("Data Distribution")
        
        # Choose visualization type
        viz_type = st.selectbox(
            "Select visualization type",
            ["Distribution Plots", "Correlation Heatmap", "Scatter Plots", "Box Plots", "Pair Plot"]
        )
        
        if viz_type == "Distribution Plots":
            # Numerical columns for histograms
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numerical_cols:
                selected_col = st.selectbox("Select column for histogram", numerical_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        data, 
                        x=selected_col, 
                        title=f"Distribution of {selected_col}",
                        color_discrete_sequence=['#FF4B4B']
                    )
                    fig.update_layout(
                        template="plotly_white",
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # KDE plot
                    fig = px.density_contour(
                        data, 
                        x=selected_col,
                        title=f"Density Plot of {selected_col}"
                    )
                    fig.update_layout(
                        template="plotly_white",
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numerical columns found for histograms.")
                
        elif viz_type == "Correlation Heatmap":
            # Correlation matrix
            numerical_data = data.select_dtypes(include=['int64', 'float64'])
            if not numerical_data.empty:
                corr = numerical_data.corr()
                
                fig = px.imshow(
                    corr, 
                    color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap"
                )
                
                fig.update_layout(
                    height=600,
                    template="plotly_white",
                    plot_bgcolor="rgba(0, 0, 0, 0)",
                    paper_bgcolor="rgba(0, 0, 0, 0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numerical columns found for correlation heatmap.")
                
        elif viz_type == "Scatter Plots":
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if len(numerical_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("Select X-axis", numerical_cols, index=0)
                
                with col2:
                    y_col = st.selectbox("Select Y-axis", [col for col in numerical_cols if col != x_col], index=0)
                
                color_col = st.selectbox("Select color variable (optional)", ["None"] + data.columns.tolist())
                
                if color_col == "None":
                    fig = px.scatter(
                        data, 
                        x=x_col, 
                        y=y_col,
                        title=f"Scatter Plot: {x_col} vs {y_col}",
                        color_discrete_sequence=['#FF4B4B']
                    )
                else:
                    fig = px.scatter(
                        data, 
                        x=x_col, 
                        y=y_col,
                        color=color_col,
                        title=f"Scatter Plot: {x_col} vs {y_col} (colored by {color_col})"
                    )
                
                fig.update_layout(
                    template="plotly_white",
                    plot_bgcolor="rgba(0, 0, 0, 0)",
                    paper_bgcolor="rgba(0, 0, 0, 0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numerical columns for scatter plots.")
                
        elif viz_type == "Box Plots":
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            
            if numerical_cols:
                y_col = st.selectbox("Select numerical variable", numerical_cols)
                
                if categorical_cols:
                    x_col = st.selectbox("Select categorical variable (optional)", ["None"] + categorical_cols)
                    
                    if x_col == "None":
                        fig = px.box(
                            data, 
                            y=y_col,
                            title=f"Box Plot of {y_col}",
                            color_discrete_sequence=['#FF4B4B']
                        )
                    else:
                        fig = px.box(
                            data, 
                            x=x_col, 
                            y=y_col,
                            title=f"Box Plot of {y_col} by {x_col}",
                            color=x_col
                        )
                else:
                    fig = px.box(
                        data, 
                        y=y_col,
                        title=f"Box Plot of {y_col}",
                        color_discrete_sequence=['#FF4B4B']
                    )
                
                fig.update_layout(
                    template="plotly_white",
                    plot_bgcolor="rgba(0, 0, 0, 0)",
                    paper_bgcolor="rgba(0, 0, 0, 0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numerical columns found for box plots.")
                
        elif viz_type == "Pair Plot":
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numerical_cols) >= 2:
                selected_cols = st.multiselect(
                    "Select columns for pair plot (max 5 recommended)",
                    numerical_cols,
                    default=numerical_cols[:min(4, len(numerical_cols))]
                )
                
                if len(selected_cols) >= 2:
                    color_col = st.selectbox("Select color variable (optional)", ["None"] + data.columns.tolist())
                    
                    if color_col == "None":
                        fig = px.scatter_matrix(
                            data,
                            dimensions=selected_cols,
                            title="Pair Plot"
                        )
                    else:
                        fig = px.scatter_matrix(
                            data,
                            dimensions=selected_cols,
                            color=color_col,
                            title=f"Pair Plot (colored by {color_col})"
                        )
                    
                    fig.update_layout(
                        height=800,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least 2 columns for the pair plot.")
            else:
                st.warning("Need at least 2 numerical columns for pair plots.")
                
        st.subheader("Target Variable Analysis")
        
        if st.session_state['target_name']:
            target_name = st.session_state['target_name']
            
            # Target distribution
            if data[target_name].dtype in ['int64', 'float64'] and data[target_name].nunique() > 10:
                # Continuous target
                fig = px.histogram(
                    data,
                    x=target_name,
                    title=f"Distribution of Target Variable: {target_name}",
                    color_discrete_sequence=['#FF4B4B']
                )
            else:
                # Categorical target
                fig = px.bar(
                    data[target_name].value_counts().reset_index(),
                    x='index',
                    y=target_name,
                    title=f"Distribution of Target Variable: {target_name}",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_xaxes(title="Categories")
                fig.update_yaxes(title="Count")
            
            fig.update_layout(
                template="plotly_white",
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (based on correlation for numerical features)
            if data[target_name].dtype in ['int64', 'float64']:
                numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if target_name in numerical_cols:
                    numerical_cols.remove(target_name)
                
                if numerical_cols:
                    correlations = data[numerical_cols].corrwith(data[target_name]).abs().sort_values(ascending=False)
                    
                    fig = px.bar(
                        correlations,
                        title=f"Feature Correlation with {target_name}",
                        color=correlations.values,
                        color_continuous_scale='viridis'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Features",
                        yaxis_title="Absolute Correlation",
                        template="plotly_white",
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# 4. MODEL TRAINING PAGE
elif selected == "Model Training":
    st.title("🧠 Model Training")
    
    if not st.session_state['preprocessed']:
        st.warning("Please preprocess your data first!")
    else:
        st.subheader("Configure and Train Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### K-Nearest Neighbors Classifier")
            knn_n_neighbors = st.slider("Number of neighbors (k)", 1, 20, 5)
            knn_weights = st.selectbox("Weight function", ["uniform", "distance"])
            knn_metric = st.selectbox("Distance metric", ["euclidean", "manhattan", "minkowski"])
            knn_p = st.slider("Power parameter for Minkowski metric", 1, 5, 2) if knn_metric == "minkowski" else 2
            
        with col2:
            st.write("### Decision Tree Classifier")
            dt_criterion = st.selectbox("Split criterion", ["gini", "entropy"])
            dt_max_depth = st.slider("Maximum depth", 1, 20, 5)
            dt_min_samples_split = st.slider("Minimum samples to split", 2, 10, 2)
            dt_min_samples_leaf = st.slider("Minimum samples in leaf", 1, 10, 1)
        
        if st.button("Train Models", key="train_button"):
            with st.spinner("Training models..."):
                # Get data from session state
                X_train = st.session_state['X_train']
                y_train = st.session_state['y_train']
                
                # Train KNN model
                knn = KNeighborsClassifier(
                    n_neighbors=knn_n_neighbors,
                    weights=knn_weights,
                    metric=knn_metric,
                    p=knn_p
                )
                knn.fit(X_train, y_train)
                st.session_state['knn_model'] = knn
                
                # Train Decision Tree model
                dt = DecisionTreeClassifier(
                    criterion=dt_criterion,
                    max_depth=dt_max_depth,
                    min_samples_split=dt_min_samples_split,
                    min_samples_leaf=dt_min_samples_leaf,
                    random_state=42
                )
                dt.fit(X_train, y_train)
                st.session_state['dt_model'] = dt
                
                st.session_state['models_trained'] = True
                
                time.sleep(1)  # Simulate training time
            
            st.success("Models trained successfully!")
            
            # Display model parameters
            st.subheader("Model Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### KNN Model")
                st.json({
                    "n_neighbors": knn_n_neighbors,
                    "weights": knn_weights,
                    "metric": knn_metric,
                    "p": knn_p
                })
            
            with col2:
                st.write("### Decision Tree Model")
                st.json({
                    "criterion": dt_criterion,
                    "max_depth": dt_max_depth,
                    "min_samples_split": dt_min_samples_split,
                    "min_samples_leaf": dt_min_samples_leaf
                })

# 5. MODEL EVALUATION PAGE
elif selected == "Model Evaluation":
    st.title("📈 Model Evaluation")
    
    if not st.session_state['models_trained']:
        st.warning("Please train your models first!")
    else:
        # Get data and models from session state
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        knn_model = st.session_state['knn_model']
        dt_model = st.session_state['dt_model']
        
        # Make predictions
        knn_preds = knn_model.predict(X_test)
        dt_preds = dt_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "Accuracy": [accuracy_score(y_test, knn_preds), accuracy_score(y_test, dt_preds)],
            "Precision": [precision_score(y_test, knn_preds, average='weighted', zero_division=0), 
                         precision_score(y_test, dt_preds, average='weighted', zero_division=0)],
            "Recall": [recall_score(y_test, knn_preds, average='weighted', zero_division=0), 
                      recall_score(y_test, dt_preds, average='weighted', zero_division=0)],
            "F1 Score": [f1_score(y_test, knn_preds, average='weighted', zero_division=0), 
                        f1_score(y_test, dt_preds, average='weighted', zero_division=0)]
        }
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics, index=["KNN", "Decision Tree"])
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        st.dataframe(metrics_df.style.highlight_max(axis=0))
        
        # Visualize metrics
        fig = go.Figure()
        
        for i, model in enumerate(["KNN", "Decision Tree"]):
            fig.add_trace(go.Bar(
                x=list(metrics.keys()),
                y=[metrics[metric][i] for metric in metrics.keys()],
                name=model,
                marker_color=['#FF9999', '#66B2FF'][i]
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### KNN Model")
            knn_cm = confusion_matrix(y_test, knn_preds)
            fig = px.imshow(
                knn_cm,
                text_auto=True,
                color_continuous_scale='blues',
                title="KNN Confusion Matrix"
            )
            fig.update_layout(
                template="plotly_white",
                width=400,
                height=400,
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            st.plotly_chart(fig)
        
        with col2:
            st.write("### Decision Tree Model")
            dt_cm = confusion_matrix(y_test, dt_preds)
            fig = px.imshow(
                dt_cm,
                text_auto=True,
                color_continuous_scale='greens',
                title="Decision Tree Confusion Matrix"
            )
            fig.update_layout(
                template="plotly_white",
                width=400,
                height=400,
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            st.plotly_chart(fig)
        
        # Model comparison conclusion
        st.subheader("Model Comparison Summary")
        
        best_model = "KNN" if metrics["Accuracy"][0] > metrics["Accuracy"][1] else "Decision Tree"
        
        st.write(f"### Best Performing Model: {best_model}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metric_card("Accuracy", f"{max(metrics['Accuracy']):.4f}")
            metric_card("Precision", f"{max(metrics['Precision']):.4f}")
        
        with col2:
            metric_card("Recall", f"{max(metrics['Recall']):.4f}")
            metric_card("F1 Score", f"{max(metrics['F1 Score']):.4f}")
        
        # Feature importance for Decision Tree
        if hasattr(dt_model, 'feature_importances_'):
            st.subheader("Feature Importance (Decision Tree)")
            
            feature_names = st.session_state['feature_names']
            importances = dt_model.feature_importances_
            
            # Create DataFrame for feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                color='Importance',
                color_continuous_scale='viridis',
                title="Feature Importance (Decision Tree)"
            )
            
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Features",
                yaxis_title="Importance",
            )
            
            st.plotly_chart(fig, use_container_width=True)

# 6. PREDICTION PAGE
elif selected == "Prediction":
    st.title("🔮 Prediction")
    
    if not st.session_state['models_trained']:
        st.warning("Please train your models first!")
    else:
        st.subheader("Make Predictions with Trained Models")
        
        # Select model
        model_option = st.radio(
            "Select model for prediction",
            ["K-Nearest Neighbors", "Decision Tree"],
            horizontal=True
        )
        
        model = st.session_state['knn_model'] if model_option == "K-Nearest Neighbors" else st.session_state['dt_model']
        
        # Get feature names and encoders
        feature_names = st.session_state['feature_names']
        categorical_columns = st.session_state['categorical_columns']
        numerical_columns = st.session_state['numerical_columns']
        encoders = st.session_state['encoders']
        scaler = st.session_state['scaler']
        
        # Method selection
        prediction_method = st.radio(
            "Choose prediction method",
            ["Input values manually", "Upload new data file"],
            horizontal=True
        )
        
        if prediction_method == "Input values manually":
            st.subheader("Enter Feature Values")
            
            # Create a dictionary to store input values
            input_data = {}
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Add input fields for each feature
            for i, feature in enumerate(feature_names):
                with col1 if i % 2 == 0 else col2:
                    if feature in categorical_columns:
                        # For categorical features, create a selectbox with unique values
                        original_values = list(encoders[feature].classes_)
                        selected_value = st.selectbox(f"{feature}", original_values)
                        input_data[feature] = selected_value
                    else:
                        # For numerical features, create a number input
                        min_val = float(st.session_state['data'][feature].min())
                        max_val = float(st.session_state['data'][feature].max())
                        step = (max_val - min_val) / 100
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val + max_val) / 2,
                            step=step
                        )
            
            if st.button("Predict", key="predict_single"):
                with st.spinner("Making prediction..."):
                    # Create a DataFrame with the input values
                    input_df = pd.DataFrame([input_data])
                    
                    # Preprocess input data
                    for col in categorical_columns:
                        input_df[col] = encoders[col].transform(input_df[col].astype(str))
                    
                    if scaler is not None and numerical_columns:
                        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Decode target if it was encoded
                    if 'target' in encoders:
                        prediction = encoders['target'].inverse_transform([prediction])[0]
                    
                    time.sleep(0.5)  # Simulate prediction time
                
                # Display prediction
                st.subheader("Prediction Result")
                
                target_name = st.session_state['target_name']
                
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #FF4B4B;">
                    <div class="metric-label">Predicted {target_name}</div>
                    <div class="metric-value" style="color: #FF4B4B; font-size: 2.5rem;">{prediction}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add confidence scores if available
                if hasattr(model, 'predict_proba'):
                    st.subheader("Prediction Confidence")
                    
                    proba = model.predict_proba(input_df)[0]
                    classes = model.classes_
                    
                    # Decode classes if target was encoded
                    if 'target' in encoders:
                        classes = encoders['target'].inverse_transform(classes)
                    
                    # Create confidence DataFrame
                    confidence_df = pd.DataFrame({
                        'Class': classes,
                        'Confidence': proba
                    }).sort_values('Confidence', ascending=False)
                    
                    # Visualize confidence
                    fig = px.bar(
                        confidence_df,
                        x='Class',
                        y='Confidence',
                        color='Confidence',
                        color_continuous_scale='RdBu',
                        title="Prediction Confidence Scores"
                    )
                    
                    fig.update_layout(
                        template="plotly_white",
                        xaxis_title="Classes",
                        yaxis_title="Confidence Score",
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:  # Upload new data file
            st.subheader("Upload New Data for Batch Prediction")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="prediction_upload")
            
            if uploaded_file is not None:
                with st.spinner('Loading data...'):
                    pred_data = pd.read_csv(uploaded_file)
                
                st.success("Data successfully loaded!")
                
                st.subheader("New Data Preview")
                st.dataframe(pred_data.head())
                
                missing_features = [feat for feat in feature_names if feat not in pred_data.columns]
                
                if missing_features:
                    st.error(f"Missing features in the dataset: {', '.join(missing_features)}")
                else:
                    # Only keep the required features
                    pred_data = pred_data[feature_names]
                    
                    if st.button("Predict", key="predict_batch"):
                        with st.spinner("Making predictions..."):
                            # Preprocess data
                            for col in categorical_columns:
                                if col in pred_data.columns:
                                    pred_data[col] = encoders[col].transform(pred_data[col].astype(str))
                            
                            if scaler is not None and numerical_columns:
                                pred_data[numerical_columns] = scaler.transform(pred_data[numerical_columns])
                            
                            # Make predictions
                            predictions = model.predict(pred_data)
                            
                            # Decode target if it was encoded
                            if 'target' in encoders:
                                predictions = encoders['target'].inverse_transform(predictions)
                            
                            # Add predictions to the data
                            pred_data_with_results = pred_data.copy()
                            pred_data_with_results[f'Predicted_{st.session_state["target_name"]}'] = predictions
                            
                            time.sleep(0.5)  # Simulate prediction time
                        
                        st.subheader("Prediction Results")
                        st.dataframe(pred_data_with_results)
                        
                        # Download button for predictions
                        csv = pred_data_with_results.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                            key="download_predictions"
                        )
                        
                        # Visualize predictions
                        st.subheader("Prediction Distribution")
                        
                        fig = px.histogram(
                            predictions,
                            title="Distribution of Predictions",
                            color_discrete_sequence=['#FF4B4B']
                        )
                        
                        fig.update_layout(
                            template="plotly_white",
                            xaxis_title=f"Predicted {st.session_state['target_name']}",
                            yaxis_title="Count",
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

# Add footer
st.markdown("""
<div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; text-align: center; margin-top: 30px;">
    <p style="color: black;">Data Science  App</p>
    <p style="font-size: 0.8rem; color: black;">Data Science Lab Experiential Learning.</p>
</div>

""", unsafe_allow_html=True)