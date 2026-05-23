import numpy as np
import matplotlib.pyplot as plt

def kmeans(X,k,max_iters=100, random_state=42):
    """
       Υλοποίηση k-means
       X: (3000, 2) (από pca)
       k: αριθμός clusters
       Επιστρέφει τα labels κάθε σημείου και τα τελικά centroids.
    """
    np.random.seed(random_state)
    #Βήμα 1, τυχαία αρχικοποίηση centroids
    indices=np.random.choice(X.shape[0], k, replace=False)
    centroids=X[indices].copy()

    labels=np.zeros(X.shape[0], dtype=int)
    for iteration in range(max_iters):
        #Βήμα 2, αναθέτω σημεία στο κοντινότερο centroid
        new_labels=assign_to_centroids(X, centroids)

        #Βήμα 3, ενημέρωση centroids με Μ.Ο κάθε cluster
        new_centroids=np.array([
            X[new_labels==i].mean(axis=0) for i in range(k)
        ])

        #Βήμα 4, έλεγχος σύγκλισης
        if np.allclose(centroids, new_centroids) and np.all(labels==new_labels):
            print(f"k-means converged after {iteration+1} iterations.")
            break
        centroids=new_centroids
        labels=new_labels

    return labels, centroids

def assign_to_centroids(X, centroids):
    """
    Για κάθε σημείο βρίσκει το κοντινότερο centroid (Ευκλείδεια απόσταση).
    """
    # Υπολογισμός αποστάσεων: (3000, k)
    distances = np.array([
        np.linalg.norm(X - c, axis=1) for c in centroids
    ]).T
    # Για κάθε σημείο, επιλέγουμε το index του κοντινότερου centroid
    return np.argmin(distances, axis=1)

def reconstruct_centroids(centroids_pca, W, mean):
    """
    Ανακατασκευή των centroids από τον χώρο PCA στον αρχικό (784 διαστάσεις).
    Αντίστροφη προβολή: X_reconstructed = centroids @ W.T + mean
    """
    return centroids_pca @ W.T + mean

def plot_kmeans(X_pca, labels, centroids, y_true, classes, k):
    """
    Διπλό plot: αριστερά τα K-means clusters, δεξιά τα αληθινά labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_clusters = ['red', 'blue', 'green']
    colors_true = ['orange', 'purple', 'cyan']

    # --- Αριστερά: K-means clusters ---
    for i in range(k):
        mask = labels == i
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors_clusters[i], label=f'Cluster {i}',
                        alpha=0.4, s=10)
    # Σχεδιασμός centroids
    axes[0].scatter(centroids[:, 0], centroids[:, 1],
                    c='black', marker='X', s=200, label='Centroids', zorder=5)
    axes[0].set_title('K-means Clusters')
    axes[0].legend()

    # --- Δεξιά: Αληθινά labels ---
    for i, c in enumerate(classes):
        mask = y_true == c
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors_true[i], label=f'Digit {c}',
                        alpha=0.4, s=10)
    axes[1].set_title('True Labels')
    axes[1].legend()

    plt.suptitle('K-means vs True Labels')
    plt.tight_layout()
    plt.show()

def run_kmeans(X_pca, y, classes, W, mean, k=3):
    # Εκτέλεση K-means
    labels, centroids = kmeans(X_pca, k=k)

    plot_kmeans(X_pca, labels, centroids, y, classes, k)
    # Ανακατασκευή centroids στις αρχικές 784 διαστάσεις
    centroids_original = reconstruct_centroids(centroids, W, mean)
    print(f"Reconstructed centroids shape: {centroids_original.shape}")  # (3, 784)
    plot_reconstructed_centroids(centroids_original, k)
    return labels, centroids, centroids_original

def plot_reconstructed_centroids(centroids_original, k):
    """
    Οπτικοποίηση της ανακατασκευής
    """
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    for i in range(k):
        # Reshape από (784,) σε (28, 28)
        img = centroids_original[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Centroid {i}')
        axes[i].axis('off')   # κρύβουμε τους άξονες - δεν χρειάζονται σε εικόνα
    plt.suptitle('Reconstructed Centroids (original space)')
    plt.tight_layout()
    plt.show()


