# Disease Classification: TF-IDF vs. One-Hot Encoding

This project compares TF-IDF feature extraction with one-hot encoding for disease classification tasks. It includes dimensionality reduction techniques and classification models to evaluate which encoding method performs better.

## Files

- `TF-IDF_vs_OneHot.ipynb`: Jupyter Notebook containing the complete implementation
- `streamlit_app.py`: Streamlit application for interactive KNN model exploration
- `disease_features (1).csv`: Dataset containing disease features as text
- `encoded_output2 (1).csv`: Dataset containing one-hot encoded features

## Tasks Implemented

### Task 1: TF-IDF Feature Extraction
- Loading and parsing the disease features dataset
- Converting risk factors, symptoms, and signs to text format
- Applying TF-IDF vectorization to each column separately
- Combining the TF-IDF matrices into a single feature matrix
- Comparing with the one-hot encoded matrix

### Task 2: Dimensionality Reduction
- Applying PCA and Truncated SVD to both matrices
- Reducing dimensions to 2-3 components
- Comparing explained variance ratios
- Visualizing the reduced dimensions in 2D plots

### Task 3: Classification Models
- Training KNN models with different k values and distance metrics
- Training Logistic Regression models
- Performing cross-validation
- Comparing performance metrics (accuracy, precision, recall, F1-score)

### Task 4: Critical Analysis
- Analyzing why TF-IDF might outperform one-hot encoding (or vice versa)
- Discussing clinical relevance of the results
- Analyzing limitations of both encoding methods

## How to Run

### Jupyter Notebook
1. Ensure you have Jupyter Notebook installed
2. Open `TF-IDF_vs_OneHot.ipynb` in Jupyter Notebook
3. Run all cells to see the complete analysis

### Streamlit Application
1. Install Streamlit if not already installed:
   ```
   pip install streamlit
   ```
2. Run the Streamlit application:
   ```
   streamlit run streamlit_app.py
   ```
3. The application will open in your web browser

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit

## Results

The notebook and Streamlit application provide a comprehensive comparison of TF-IDF and one-hot encoding for disease classification. Key findings include:

1. Dimensionality reduction effectiveness for each encoding method
2. Classification performance across different models and parameters
3. Visualization of disease clusters in reduced dimensional space
4. Analysis of which encoding method is more suitable for clinical applications

## Conclusion

Based on the analysis, we can determine which encoding method is more suitable for disease classification tasks and why. The results provide insights into the trade-offs between TF-IDF and one-hot encoding in the context of medical data analysis.
