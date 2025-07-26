# Olivetti Faces Classification – face recognition using the Olivetti Faces dataset

Welcome to the **face recognition using the Olivetti Faces dataset** project!  
This repository provides a professional, end-to-end workflow for face recognition using the Olivetti Faces dataset, implemented in a Jupyter Notebook with clear explanations and modern machine learning best practices.

---

## Overview

This project demonstrates how to perform facial recognition and analysis on the Olivetti Faces dataset using Python and scikit-learn. The notebook guides you through:

- **Data exploration & visualization**: Understand the dataset, visualize faces and target distribution.
- **Dimensionality reduction**: Use PCA (Principal Component Analysis) to extract Eigenfaces and reduce feature space.
- **Model training & evaluation**: Compare multiple classifiers (SVM, Logistic Regression, LDA, etc.) with robust metrics.
- **Cross-validation & hyperparameter tuning**: Ensure accuracy and generalization using k-fold, leave-one-out CV, and grid search.
- **Advanced metrics**: Precision-recall curves for multi-class analysis.
- **Modular, pipeline-based workflow**: Professional, reusable code for experiments and further research.

---

## Features

- 📊 **Exploratory Data Analysis**: Visualize all faces, sample distributions, and PCA projections.
- 💡 **Dimensionality Reduction**: Extract and visualize Eigenfaces, analyze explained variance.
- 🤖 **Classifier Comparison**: Train, validate, and compare SVM, Logistic Regression, LDA, KNN, Decision Tree, and Naive Bayes.
- 🏆 **Performance Metrics**: Accuracy, confusion matrix, classification report, precision-recall curves.
- 🔬 **Cross-Validation**: 5-fold and leave-one-out for robust model selection.
- ⚙️ **Hyperparameter Tuning**: GridSearchCV for optimal SVM parameters.
- 🔁 **ML Pipelines**: Example of combining LDA with Logistic Regression for streamlined workflows.

---

## Getting Started

**Requirements**

- Python 3.11+
- Jupyter Notebook
- scikit-learn
- numpy, pandas, matplotlib, seaborn

**Setup**

1. Clone the repository:
    ```bash
    git clone https://github.com/youssef-ben-letaifa/Olivetti-Faces-Classification-Enhanced.git
    cd Olivetti-Faces-Classification-Enhanced
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Or install manually: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`)*

3. Download the Olivetti Faces data 

4. Launch Jupyter Notebook and open `Olivetti_Faces_Classification_Enhanced.ipynb`.

---

## Usage

- Run the notebook cell by cell to explore, train, and evaluate models.
- Modify parameters, models, or workflows as desired to experiment further.
- Visualize results and gain insights into facial recognition techniques.

---

## Results

- Achieve high accuracy on test data using PCA-reduced features and tuned classifiers.
- Obtain detailed performance comparisons and visualizations for each model.
- Understand the impact of dimensionality reduction and cross-validation on face recognition.

---
