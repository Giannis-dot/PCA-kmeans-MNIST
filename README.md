PCA, K-means & Mahalanobis Classifier on MNIST

Overview
Implementation of dimensionality reduction, clustering and classification 
algorithms on the MNIST handwritten digits dataset, built from scratch using NumPy.

Algorithms Implemented

PCA (Principal Component Analysis)
Dimensionality reduction from 784 to 2 or 10 dimensions. 
Implemented from scratch using covariance matrix eigendecomposition.

K-means Clustering
Unsupervised clustering in the reduced PCA space.
Implemented from scratch with random centroid initialization and iterative updates.
Cluster centroids are reconstructed back to the original 784-dimensional space.

Mahalanobis Classifier
Nearest centroid classifier using Mahalanobis distance, which accounts 
for the shape and spread of each class. Two approaches are compared:
- 10 classes: one centroid per digit → 87.14% accuracy
- 20 subclasses: two centroids per digit via K-means → 88.65% accuracy

Installation
pip install numpy matplotlib datasets

Usage
python main.py
