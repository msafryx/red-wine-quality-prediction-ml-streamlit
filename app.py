import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.sub-header {
    font-size: 1.5rem;
    color: #2c5282;
    margin: 1rem 0;
}
.metric-container {
    background-color: #f7fafc;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4299e1;
    margin: 0.5rem 0;
}
.prediction-result {
    font-size: 1.2rem;
    font-weight: bold;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin: 1rem 0;
}
.survived {
    background-color: #c6f6d5;
    color: #22543d;
    border: 2px solid #48bb78;
}
.not-survived {
    background-color: #fed7d7;
    color: #742a2a;
    border: 2px solid #e53e3e;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the Titanic dataset"""
    try:
        # Try to load from local file first
        data = pd.read_csv('data/titanic.csv')
    except:
        # If local file doesn't exist, create sample data for demonstration
        st.warning("Dataset file not found. Using sample data for demonstration.")
        # Create sample Titanic data
        np.random.seed(42)
        n_samples = 891
        
        data = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Name': [f"Passenger {i}" for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(29, 14, n_samples).clip(0, 80),
            'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.01, 0.003]),
            'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.02, 0.006, 0.002, 0.001]),
            'Ticket': [f"TICKET{i}" for i in range(1, n_samples + 1)],
            'Fare': np.random.lognormal(3, 1, n_samples).clip(0, 512),
            'Cabin': np.random.choice(['A1', 'B2', 'C3', None], n_samples, p=[0.1, 0.1, 0.1, 0.7]),
            'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
        })
        
        # Add some missing values to Age
        missing_age_indices = np.random.choice(data.index, size=int(0.2 * len(data)), replace=False)
        data.loc[missing_age_indices, 'Age'] = np.nan
    
    return data

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Create a copy
    data = df.copy()
    
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Feature engineering
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, np.inf], 
                              labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    data['Sex_encoded'] = le_sex.fit_transform(data['Sex'])
    
    le_embarked = LabelEncoder()
    data['Embarked_encoded'] = le_embarked.fit_transform(data['Embarked'])
    
    # Create dummy variables for Pclass
    pclass_dummies = pd.get_dummies(data['Pclass'], prefix='Pclass')
    data = pd.concat([data, pclass_dummies], axis=1)
    
    return data, le_sex, le_embarked

@st.cache_data
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    return results, trained_models

def main():
    # Main title
    st.markdown('<h1 class="main-header">🚢 Titanic Survival Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📊 Data Exploration", "📈 Data Visualization", "🤖 Model Training", "🎯 Make Predictions", "📋 Model Performance"]
    )
    
    # Load and preprocess data
    raw_data = load_data()
    processed_data, le_sex, le_embarked = preprocess_data(raw_data)
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Titanic Survival Prediction System!
        
        This interactive web application analyzes the famous Titanic dataset and predicts passenger survival 
        using advanced machine learning algorithms.
        
        ### 🎯 Project Overview
        The RMS Titanic sank on April 15, 1912, during its maiden voyage. This tragedy resulted in the deaths 
        of 1502 out of 2224 passengers and crew. This application uses passenger data to predict survival chances 
        based on various factors such as age, gender, ticket class, and family relationships.
        
        ### 🚀 Features
        - **Comprehensive Data Exploration**: Dive deep into the dataset with interactive filtering
        - **Rich Visualizations**: Multiple charts and plots to understand survival patterns  
        - **Machine Learning Models**: Compare performance of different algorithms
        - **Real-time Predictions**: Input passenger details and get instant survival predictions
        - **Model Performance Analysis**: Detailed metrics and comparison charts
        
        ### 📱 How to Use
        Navigate through different sections using the sidebar to explore the data, understand the models, 
        and make your own predictions!
        
        ### 📊 Dataset Information
        - **Total Passengers**: {:,}
        - **Features**: {} 
        - **Survival Rate**: {:.1f}%
        """.format(
            len(raw_data),
            len(raw_data.columns),
            raw_data['Survived'].mean() * 100
        ))
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Passengers", f"{len(raw_data):,}")
        with col2:
            st.metric("Survived", f"{raw_data['Survived'].sum():,}")
        with col3:
            st.metric("Survival Rate", f"{raw_data['Survived'].mean():.1%}")
        with col4:
            st.metric("Average Age", f"{raw_data['Age'].mean():.1f} years")
    
    elif page == "📊 Data Exploration":
        st.markdown('<h2 class="sub-header">Data Exploration</h2>', unsafe_allow_html=True)
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Shape:**", raw_data.shape)
            st.write("**Missing Values:**")
            missing_data = raw_data.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                st.dataframe(missing_data.to_frame('Missing Count'))
            else:
                st.write("No missing values!")
        
        with col2:
            st.write("**Data Types:**")
            st.dataframe(raw_data.dtypes.to_frame('Data Type'))
        
        # Sample data
        st.subheader("🔍 Sample Data")
        st.dataframe(raw_data.head(10))
        
        # Interactive filtering
        st.subheader("🎛️ Interactive Data Filtering")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            survival_filter = st.selectbox("Survival Status", ["All", "Survived", "Did not survive"])
            gender_filter = st.selectbox("Gender", ["All"] + list(raw_data['Sex'].unique()))
        
        with col2:
            class_filter = st.selectbox("Passenger Class", ["All"] + [f"Class {i}" for i in sorted(raw_data['Pclass'].unique())])
            age_range = st.slider("Age Range", 0, int(raw_data['Age'].max()), (0, int(raw_data['Age'].max())))
        
        with col3:
            embarked_filter = st.selectbox("Port of Embarkation", ["All"] + list(raw_data['Embarked'].dropna().unique()))
        
        # Apply filters
        filtered_data = raw_data.copy()
        
        if survival_filter != "All":
            filtered_data = filtered_data[filtered_data['Survived'] == (1 if survival_filter == "Survived" else 0)]
        if gender_filter != "All":
            filtered_data = filtered_data[filtered_data['Sex'] == gender_filter]
        if class_filter != "All":
            class_num = int(class_filter.split()[-1])
            filtered_data = filtered_data[filtered_data['Pclass'] == class_num]
        if embarked_filter != "All":
            filtered_data = filtered_data[filtered_data['Embarked'] == embarked_filter]
        
        filtered_data = filtered_data[(filtered_data['Age'] >= age_range[0]) & (filtered_data['Age'] <= age_range[1])]
        
        st.write(f"**Filtered Results**: {len(filtered_data)} passengers")
        st.dataframe(filtered_data)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        st.dataframe(filtered_data.describe())
    
    elif page == "📈 Data Visualization":
        st.markdown('<h2 class="sub-header">Data Visualization</h2>', unsafe_allow_html=True)
        
        # Survival by Gender
        st.subheader("👥 Survival Analysis by Gender")
        fig1 = px.histogram(raw_data, x='Sex', color='Survived', 
                           title='Survival Count by Gender',
                           labels={'Survived': 'Survival Status', 'count': 'Number of Passengers'},
                           color_discrete_map={0: 'red', 1: 'green'})
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Age distribution
        st.subheader("📊 Age Distribution and Survival")
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = px.histogram(raw_data, x='Age', color='Survived', 
                               title='Age Distribution by Survival',
                               nbins=30,
                               color_discrete_map={0: 'red', 1: 'green'})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fig3 = px.box(raw_data, x='Survived', y='Age', 
                         title='Age Distribution by Survival Status',
                         color='Survived',
                         color_discrete_map={0: 'red', 1: 'green'})
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Passenger Class Analysis
        st.subheader("🎭 Survival by Passenger Class")
        survival_by_class = raw_data.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
        survival_rate_by_class = raw_data.groupby('Pclass')['Survived'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.bar(x=survival_rate_by_class.index, y=survival_rate_by_class.values,
                         title='Survival Rate by Passenger Class',
                         labels={'x': 'Passenger Class', 'y': 'Survival Rate'},
                         color=survival_rate_by_class.values,
                         color_continuous_scale='RdYlGn')
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            fig5 = px.sunburst(raw_data, path=['Pclass', 'Sex', 'Survived'], 
                              title='Hierarchical View: Class → Gender → Survival')
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
        
        # Correlation Heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        corr_matrix = raw_data[numeric_cols].corr()
        
        fig6 = px.imshow(corr_matrix, 
                        title='Correlation Matrix of Numeric Features',
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
        fig6.update_layout(height=500)
        st.plotly_chart(fig6, use_container_width=True)
    
    elif page == "🤖 Model Training":
        st.markdown('<h2 class="sub-header">Model Training & Comparison</h2>', unsafe_allow_html=True)
        
        # Prepare features
        feature_cols = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded', 'FamilySize', 'IsAlone']
        X = processed_data[feature_cols]
        y = processed_data['Survived']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.write("### 📊 Training Data Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        with col3:
            st.metric("Features", len(feature_cols))
        
        # Train models
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a moment."):
                results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.trained_models = trained_models
                st.session_state.scaler = scaler
                st.session_state.feature_cols = feature_cols
                st.session_state.le_sex = le_sex
                st.session_state.le_embarked = le_embarked
        
        # Display results if available
        if 'results' in st.session_state:
            st.success("✅ Models trained successfully!")
            
            # Model comparison
            st.subheader("🏆 Model Performance Comparison")
            
            model_metrics = []
            for name, result in st.session_state.results.items():
                model_metrics.append({
                    'Model': name,
                    'Accuracy': f"{result['accuracy']:.3f}",
                    'CV Mean': f"{result['cv_mean']:.3f}",
                    'CV Std': f"{result['cv_std']:.3f}"
                })
            
            metrics_df = pd.DataFrame(model_metrics)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model
            best_model_name = max(st.session_state.results.keys(), 
                                key=lambda x: st.session_state.results[x]['accuracy'])
            st.success(f"🥇 Best Model: **{best_model_name}** with accuracy: {st.session_state.results[best_model_name]['accuracy']:.3f}")
            
            # Feature importance (for Random Forest)
            if 'Random Forest' in st.session_state.results:
                st.subheader("📊 Feature Importance (Random Forest)")
                rf_model = st.session_state.results['Random Forest']['model']
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Make Predictions":
        st.markdown('<h2 class="sub-header">Make Survival Predictions</h2>', unsafe_allow_html=True)
        
        if 'trained_models' not in st.session_state:
            st.warning("⚠️ Please train the models first by visiting the Model Training section.")
            return
        
        st.write("### 👤 Enter Passenger Information")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First Class, 2 = Second Class, 3 = Third Class")
            sex = st.selectbox("Gender", ['male', 'female'])
            age = st.number_input("Age", min_value=0, max_value=100, value=30, help="Age in years")
            sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=10, value=0)
        
        with col2:
            parch = st.number_input("Number of Parents/Children", min_value=0, max_value=10, value=0)
            fare = st.number_input("Ticket Fare", min_value=0.0, max_value=1000.0, value=50.0, help="Fare in dollars")
            embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'], 
                                  help="C = Cherbourg, Q = Queenstown, S = Southampton")
        
        # Model selection
        model_choice = st.selectbox("Choose Model", list(st.session_state.trained_models.keys()))
        
        if st.button("🔮 Predict Survival", type="primary"):
            # Prepare input data
            sex_encoded = st.session_state.le_sex.transform([sex])[0]
            embarked_encoded = st.session_state.le_embarked.transform([embarked])[0]
            family_size = sibsp + parch + 1
            is_alone = 1 if family_size == 1 else 0
            
            # Create input array
            input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded, family_size, is_alone]])
            
            # Scale input data
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Make prediction
            model = st.session_state.trained_models[model_choice]
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.write("### 🎯 Prediction Results")
            
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-result survived">✅ SURVIVED<br>Survival Probability: {prediction_proba[1]:.1%}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-result not-survived">❌ DID NOT SURVIVE<br>Survival Probability: {prediction_proba[1]:.1%}</div>',
                    unsafe_allow_html=True
                )
            
            # Probability breakdown
            st.write("### 📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': ['Did not survive', 'Survived'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            
            fig = px.bar(prob_df, x='Outcome', y='Probability', 
                        title=f'Survival Probabilities ({model_choice})',
                        color='Probability',
                        color_continuous_scale='RdYlGn')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Passenger profile summary
            st.write("### 👤 Passenger Profile Summary")
            profile_data = {
                'Attribute': ['Class', 'Gender', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Embarkation Port', 'Family Size', 'Traveling Alone'],
                'Value': [f"Class {pclass}", sex.title(), f"{age} years", sibsp, parch, f"${fare:.2f}", 
                         {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}[embarked],
                         family_size, 'Yes' if is_alone else 'No']
            }
            st.dataframe(pd.DataFrame(profile_data), use_container_width=True)
    
    elif page == "📋 Model Performance":
        st.markdown('<h2 class="sub-header">Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Please train the models first by visiting the Model Training section.")
            return
        
        # Model selection for detailed analysis
        selected_model = st.selectbox("Select Model for Detailed Analysis", 
                                    list(st.session_state.results.keys()))
        
        result = st.session_state.results[selected_model]
        
        # Performance metrics
        st.subheader(f"📊 {selected_model} Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{result['accuracy']:.3f}")
        with col2:
            st.metric("CV Mean", f"{result['cv_mean']:.3f}")
        with col3:
            st.metric("CV Std", f"{result['cv_std']:.3f}")
        
        # Classification report
        st.subheader("📈 Detailed Classification Report")
        report_df = pd.DataFrame(result['classification_report']).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
        
        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        
        # Prepare data for confusion matrix
        feature_cols = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded', 'FamilySize', 'IsAlone']
        X = processed_data[feature_cols]
        y = processed_data['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        cm = confusion_matrix(y_test, result['predictions'])
        
        # Create confusion matrix heatmap
        fig = px.imshow(cm, 
                       title=f'Confusion Matrix - {selected_model}',
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Did not survive', 'Survived'],
                       y=['Did not survive', 'Survived'],
                       color_continuous_scale='Blues',
                       text_auto=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison chart
        st.subheader("🏆 All Models Comparison")
        
        comparison_data = []
        for name, res in st.session_state.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': res['accuracy'],
                'CV_Mean': res['cv_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison chart
        fig = go.Figure(data=[
            go.Bar(name='Test Accuracy', x=comparison_df['Model'], y=comparison_df['Accuracy']),
            go.Bar(name='CV Mean', x=comparison_df['Model'], y=comparison_df['CV_Mean'])
        ])
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()