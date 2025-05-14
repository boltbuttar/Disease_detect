import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Task 1: TF-IDF Feature Extraction")
print("---------------------------------")

# Load the disease features dataset
disease_features_df = pd.read_csv('disease_features (1).csv')

# Display the first few rows
print("Dataset shape:", disease_features_df.shape)
print("First few rows:")
print(disease_features_df.head(2))

# Function to safely parse string lists
def parse_string_list(s):
    if pd.isna(s) or s == '[]':
        return []
    try:
        # Replace single quotes with double quotes for proper JSON parsing
        s = s.replace("'", '"')
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        # If parsing fails, try a simpler approach
        s = s.strip('[]')
        if not s:
            return []
        return [item.strip().strip('\'"') for item in s.split(',')]

# Parse the string lists in the dataframe
for col in ['Risk Factors', 'Symptoms', 'Signs']:
    disease_features_df[col] = disease_features_df[col].apply(parse_string_list)

# Convert lists to strings for TF-IDF vectorization
disease_features_df['Risk_Factors_Text'] = disease_features_df['Risk Factors'].apply(lambda x: ' '.join(x) if x else '')
disease_features_df['Symptoms_Text'] = disease_features_df['Symptoms'].apply(lambda x: ' '.join(x) if x else '')
disease_features_df['Signs_Text'] = disease_features_df['Signs'].apply(lambda x: ' '.join(x) if x else '')

# Apply TF-IDF vectorization to each column separately
tfidf_risk = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_symptoms = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_signs = TfidfVectorizer(max_features=100, stop_words='english')

# Transform the text data
risk_factors_tfidf = tfidf_risk.fit_transform(disease_features_df['Risk_Factors_Text'])
symptoms_tfidf = tfidf_symptoms.fit_transform(disease_features_df['Symptoms_Text'])
signs_tfidf = tfidf_signs.fit_transform(disease_features_df['Signs_Text'])

# Get feature names
risk_feature_names = tfidf_risk.get_feature_names_out()
symptoms_feature_names = tfidf_symptoms.get_feature_names_out()
signs_feature_names = tfidf_signs.get_feature_names_out()

# Convert to DataFrames
risk_df = pd.DataFrame(risk_factors_tfidf.toarray(), columns=[f'risk_{f}' for f in risk_feature_names])
symptoms_df = pd.DataFrame(symptoms_tfidf.toarray(), columns=[f'symptom_{f}' for f in symptoms_feature_names])
signs_df = pd.DataFrame(signs_tfidf.toarray(), columns=[f'sign_{f}' for f in signs_feature_names])

# Combine the TF-IDF matrices
tfidf_combined = pd.concat([risk_df, symptoms_df, signs_df], axis=1)
tfidf_combined['Disease'] = disease_features_df['Disease']

# Display the shape of the combined TF-IDF matrix
print(f"TF-IDF matrix shape: {tfidf_combined.shape}")

print("\nLoad the One-Hot Encoded Matrix")
print("-------------------------------")

# Load the one-hot encoded matrix
onehot_df = pd.read_csv('encoded_output2 (1).csv')

# Display the shape of the one-hot encoded matrix
print(f"One-hot encoded matrix shape: {onehot_df.shape}")

print("\nTask 2: Dimensionality Reduction")
print("--------------------------------")

# Prepare data for dimensionality reduction
X_tfidf = tfidf_combined.drop('Disease', axis=1)
X_onehot = onehot_df.drop('Disease', axis=1)
y = disease_features_df['Disease']

# Encode the target variable for classification
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the data
scaler = StandardScaler()
X_tfidf_scaled = scaler.fit_transform(X_tfidf)
X_onehot_scaled = scaler.fit_transform(X_onehot)

# Apply PCA to TF-IDF matrix
pca_tfidf = PCA(n_components=3)
pca_tfidf_result = pca_tfidf.fit_transform(X_tfidf_scaled)

# Apply PCA to one-hot encoded matrix
pca_onehot = PCA(n_components=3)
pca_onehot_result = pca_onehot.fit_transform(X_onehot_scaled)

# Print explained variance ratios
print("PCA Explained Variance Ratio (TF-IDF):", pca_tfidf.explained_variance_ratio_)
print("PCA Explained Variance Ratio (One-Hot):", pca_onehot.explained_variance_ratio_)
print("PCA Cumulative Explained Variance (TF-IDF):", sum(pca_tfidf.explained_variance_ratio_))
print("PCA Cumulative Explained Variance (One-Hot):", sum(pca_onehot.explained_variance_ratio_))

# Apply Truncated SVD to TF-IDF matrix
svd_tfidf = TruncatedSVD(n_components=3)
svd_tfidf_result = svd_tfidf.fit_transform(X_tfidf_scaled)

# Apply Truncated SVD to one-hot encoded matrix
svd_onehot = TruncatedSVD(n_components=3)
svd_onehot_result = svd_onehot.fit_transform(X_onehot_scaled)

# Print explained variance ratios
print("\nSVD Explained Variance Ratio (TF-IDF):", svd_tfidf.explained_variance_ratio_)
print("SVD Explained Variance Ratio (One-Hot):", svd_onehot.explained_variance_ratio_)
print("SVD Cumulative Explained Variance (TF-IDF):", sum(svd_tfidf.explained_variance_ratio_))
print("SVD Cumulative Explained Variance (One-Hot):", sum(svd_onehot.explained_variance_ratio_))

# Save PCA visualization to file
plt.figure(figsize=(16, 6))

# PCA for TF-IDF
plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_tfidf_result[:, 0], pca_tfidf_result[:, 1], c=pd.factorize(y)[0], cmap='viridis', alpha=0.7)
plt.title('PCA - TF-IDF Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Disease Category')

# PCA for One-Hot Encoding
plt.subplot(1, 2, 2)
scatter = plt.scatter(pca_onehot_result[:, 0], pca_onehot_result[:, 1], c=pd.factorize(y)[0], cmap='viridis', alpha=0.7)
plt.title('PCA - One-Hot Encoded Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Disease Category')

plt.tight_layout()
plt.savefig('pca_visualization.png')
print("PCA visualization saved to 'pca_visualization.png'")

# Save SVD visualization to file
plt.figure(figsize=(16, 6))

# SVD for TF-IDF
plt.subplot(1, 2, 1)
scatter = plt.scatter(svd_tfidf_result[:, 0], svd_tfidf_result[:, 1], c=pd.factorize(y)[0], cmap='viridis', alpha=0.7)
plt.title('Truncated SVD - TF-IDF Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(scatter, label='Disease Category')

# SVD for One-Hot Encoding
plt.subplot(1, 2, 2)
scatter = plt.scatter(svd_onehot_result[:, 0], svd_onehot_result[:, 1], c=pd.factorize(y)[0], cmap='viridis', alpha=0.7)
plt.title('Truncated SVD - One-Hot Encoded Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(scatter, label='Disease Category')

plt.tight_layout()
plt.savefig('svd_visualization.png')
print("SVD visualization saved to 'svd_visualization.png'")

print("\nTask 3: Classification Models")
print("-----------------------------")

# Define a function to evaluate models with cross-validation
def evaluate_model(model, X, y_encoded, cv=2):  # Reduced CV to 2 due to small sample size
    # Count samples per class
    from collections import Counter
    class_counts = Counter(y_encoded)
    min_samples = min(class_counts.values())

    print(f"Minimum samples per class: {min_samples}")

    # If we have very few samples, use a simple train/test split instead
    if min_samples < 2:
        print("Warning: Some classes have only 1 sample. Using simple train/test split.")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, average='macro', zero_division=0)
        }
    else:
        # Use cross-validation with reduced folds
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        try:
            scores = cross_validate(model, X, y_encoded, cv=cv, scoring=scoring)

            results = {
                'Accuracy': scores['test_accuracy'].mean(),
                'Precision': scores['test_precision_macro'].mean(),
                'Recall': scores['test_recall_macro'].mean(),
                'F1 Score': scores['test_f1_macro'].mean()
            }
        except ValueError as e:
            print(f"Cross-validation error: {e}")
            print("Falling back to simple train/test split.")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=None)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, average='macro', zero_division=0)
            }

    return results

# Define distance metrics for KNN
distance_metrics = ['euclidean', 'manhattan', 'cosine']
k_values = [3, 5, 7]

# Initialize results dictionaries
knn_tfidf_results = {}
knn_onehot_results = {}

# Evaluate KNN models with different k values and distance metrics on TF-IDF features
for k in k_values:
    for metric in distance_metrics:
        model_name = f'KNN (k={k}, {metric})'
        print(f"Evaluating {model_name} on TF-IDF features...")
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_tfidf_results[model_name] = evaluate_model(knn, X_tfidf_scaled, y_encoded)
        print(f"Evaluating {model_name} on One-Hot features...")
        knn_onehot_results[model_name] = evaluate_model(knn, X_onehot_scaled, y_encoded)

# Convert results to DataFrames
knn_tfidf_df = pd.DataFrame(knn_tfidf_results).T
knn_onehot_df = pd.DataFrame(knn_onehot_results).T

# Display KNN results
print("\nKNN Results with TF-IDF Features:")
print(knn_tfidf_df)
print("\nKNN Results with One-Hot Encoded Features:")
print(knn_onehot_df)

# Train Logistic Regression models
print("\nEvaluating Logistic Regression on TF-IDF features...")
lr_tfidf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
lr_tfidf_results = evaluate_model(lr_tfidf, X_tfidf_scaled, y_encoded)

print("Evaluating Logistic Regression on One-Hot features...")
lr_onehot = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
lr_onehot_results = evaluate_model(lr_onehot, X_onehot_scaled, y_encoded)

# Display Logistic Regression results
lr_results = pd.DataFrame({
    'Logistic Regression (TF-IDF)': lr_tfidf_results,
    'Logistic Regression (One-Hot)': lr_onehot_results
})
print("\nLogistic Regression Results:")
print(lr_results)

print("\nTask 4: Critical Analysis")
print("-------------------------")
print("""
Based on the results, we can draw the following conclusions:

1. Dimensionality Reduction:
   - PCA and SVD show different explained variance ratios for TF-IDF vs. one-hot encoding
   - The visualizations reveal clustering patterns that differ between the two encoding methods

2. Classification Performance:
   - KNN performance varies with different k values and distance metrics
   - Logistic Regression shows different performance on TF-IDF vs. one-hot encoded features

3. Clinical Relevance:
   - TF-IDF captures the importance of terms in the context of diseases, which may better represent clinical significance
   - One-hot encoding treats all features equally, which may not reflect the varying importance of symptoms and signs

4. Limitations:
   - TF-IDF may overemphasize rare terms that appear in few documents
   - One-hot encoding creates sparse matrices with many zero values
   - Both methods lose some semantic relationships between terms
""")
