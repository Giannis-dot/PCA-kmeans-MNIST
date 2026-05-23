import numpy as np
import matplotlib.pyplot as plt

def pca(X, components=2):
    #Step 1, centering
    mean=np.mean(X, axis=0)
    X_centered=X-mean

    #Step 2, covariance matrix
    n=X_centered.shape[0]
    cov_matrix=X_centered.T @ X_centered / (n - 1)

    #Step 3, eigen decomposition
    eigenvalues, eigenvectors=np.linalg.eigh(cov_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    #Step 4, k principal components
    W=eigenvectors[:, :components]

    #Step 5, projection
    X_pca=X_centered @ W

    variance_explained=np.sum(eigenvalues[:components]) / np.sum(eigenvalues) * 100
    print(f"PCA: {components} components explain {variance_explained:.1f}% of variance.")
    return X_pca, W, mean

def plot_pca(X_pca, y, classes, components):
    colors = ['red', 'blue', 'green']
    fig = plt.figure(figsize=(8, 6))

    if components==2:
        ax = fig.add_subplot(111)
        for i, c in enumerate(classes):
            mask = y == c
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[i], label=f'Digit {c}',
                       alpha=0.4, s=10)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')

    elif components==3:
        ax=fig.add_subplot(111, projection='3d')
        for i, c in enumerate(classes):
            mask = y == c
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                       c=colors[i], label=f'Digit {c}',
                       alpha=0.4, s=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    plt.title(f'PCA - {components}D projection of digits {classes}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_pca(X, y, classes, components=2):
    X_pca, W, mean=pca(X, components)
    plot_pca(X_pca, y, classes, components)
    return X_pca, W, mean





