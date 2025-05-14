import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Disease Classification with KNN", layout="wide")

st.title("Disease Classification using KNN")
st.markdown("""
This application demonstrates the use of K-Nearest Neighbors (KNN) for disease classification
using both TF-IDF and One-Hot Encoded features.
""")

@st.cache_data
def parse_string_list(s):
    if pd.isna(s) or s == '[]':
        return []
    try:
        s = s.replace("'", '"')
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        # If parsing fails, try a simpler approach
        s = s.strip('[]')
        if not s:
            return []
        return [item.strip().strip('\'"') for item in s.split(',')]

@st.cache_data
def load_data():
    disease_features_df = pd.read_csv('disease_features (1).csv')
    onehot_df = pd.read_csv('encoded_output2 (1).csv')

    for col in ['Risk Factors', 'Symptoms', 'Signs']:
        disease_features_df[col] = disease_features_df[col].apply(parse_string_list)

    disease_features_df['Risk_Factors_Text'] = disease_features_df['Risk Factors'].apply(lambda x: ' '.join(x) if x else '')
    disease_features_df['Symptoms_Text'] = disease_features_df['Symptoms'].apply(lambda x: ' '.join(x) if x else '')
    disease_features_df['Signs_Text'] = disease_features_df['Signs'].apply(lambda x: ' '.join(x) if x else '')

    return disease_features_df, onehot_df

@st.cache_data
def apply_tfidf(disease_features_df):
    tfidf_risk = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_symptoms = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_signs = TfidfVectorizer(max_features=100, stop_words='english')

    risk_factors_tfidf = tfidf_risk.fit_transform(disease_features_df['Risk_Factors_Text'])
    symptoms_tfidf = tfidf_symptoms.fit_transform(disease_features_df['Symptoms_Text'])
    signs_tfidf = tfidf_signs.fit_transform(disease_features_df['Signs_Text'])

    risk_feature_names = tfidf_risk.get_feature_names_out()
    symptoms_feature_names = tfidf_symptoms.get_feature_names_out()
    signs_feature_names = tfidf_signs.get_feature_names_out()

    # Convert to DataFrames
    risk_df = pd.DataFrame(risk_factors_tfidf.toarray(), columns=[f'risk_{f}' for f in risk_feature_names])
    symptoms_df = pd.DataFrame(symptoms_tfidf.toarray(), columns=[f'symptom_{f}' for f in symptoms_feature_names])
    signs_df = pd.DataFrame(signs_tfidf.toarray(), columns=[f'sign_{f}' for f in signs_feature_names])

    # Combine the TF-IDF matrices
    tfidf_combined = pd.concat([risk_df, symptoms_df, signs_df], axis=1)

    return tfidf_combined, tfidf_risk, tfidf_symptoms, tfidf_signs

# Train KNN model
@st.cache_data
def train_knn_model(X, y, k, metric):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode the target variable
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Convert encoded predictions back to original labels for display
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    return knn, scaler, X_train, X_test, y_train, y_test, y_pred, accuracy, precision, recall, f1, cm, y_test_labels, y_pred_labels, label_encoder

# Load data
disease_features_df, onehot_df = load_data()

st.sidebar.header("Model Parameters")
encoding_method = st.sidebar.selectbox("Encoding Method", ["TF-IDF", "One-Hot"])
k_value = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
distance_metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"])

if encoding_method == "TF-IDF":
    tfidf_combined, _, _, _ = apply_tfidf(disease_features_df)
    X = tfidf_combined
    st.sidebar.info(f"TF-IDF Features: {X.shape[1]}")
else:  # One-Hot
    X = onehot_df.drop('Disease', axis=1)
    st.sidebar.info(f"One-Hot Features: {X.shape[1]}")

y = disease_features_df['Disease']

# Train model and get results
knn, scaler, X_train, X_test, y_train, y_test, y_pred, accuracy, precision, recall, f1, cm, y_test_labels, y_pred_labels, label_encoder = train_knn_model(X, y, k_value, distance_metric)

st.header("Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(10, 8))
class_names = label_encoder.classes_
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(len(class_names)), yticklabels=range(len(class_names)))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
st.pyplot(fig)

# Dimensionality reduction for visualization
st.header("Dimensionality Reduction Visualization")
reduction_method = st.selectbox("Reduction Method", ["PCA", "Truncated SVD"])

if reduction_method == "PCA":
    reducer = PCA(n_components=2)
else:  # Truncated SVD
    reducer = TruncatedSVD(n_components=2)

# Apply dimensionality reduction
X_scaled = scaler.transform(X)
X_reduced = reducer.fit_transform(X_scaled)

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=pd.factorize(y)[0], cmap='viridis', alpha=0.7)
plt.title(f'{reduction_method} - {encoding_method} Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(scatter, label='Disease Category')
st.pyplot(fig)

# Display sample predictions
st.header("Sample Predictions")
sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
sample_data = pd.DataFrame({
    'True Disease': y_test_labels[sample_indices],
    'Predicted Disease': y_pred_labels[sample_indices]
})
st.table(sample_data)

# Comparison between TF-IDF and One-Hot
st.header("Analysis: TF-IDF vs. One-Hot Encoding")
st.markdown("""
### Key Differences:

1. **Feature Representation**:
   - **TF-IDF**: Weights terms based on their frequency in a document and rarity across documents
   - **One-Hot**: Binary representation (presence/absence) of features

2. **Information Capture**:
   - **TF-IDF**: Captures the importance of terms in context
   - **One-Hot**: Treats all features equally regardless of importance

3. **Dimensionality**:
   - **TF-IDF**: Can control dimensionality with parameters like max_features
   - **One-Hot**: Often results in very high-dimensional sparse matrices

4. **Clinical Relevance**:
   - **TF-IDF**: May better represent the varying importance of symptoms and signs
   - **One-Hot**: May miss the relative importance of clinical features

Try different parameters to see how they affect model performance!
""")
