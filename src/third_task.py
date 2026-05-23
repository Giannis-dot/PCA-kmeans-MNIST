import numpy as np
from src.pca_impl import pca
from src.k_means import kmeans
import matplotlib.pyplot as plt
def compute_class_stats(X, labels, n_classes):
    """
    Υπολογίζει covariance matrix για κάθε κλάση
    """
    means=[]
    covariances=[]

    for c in range(n_classes):
        X_c=X[labels==c]
        mean_c=np.mean(X_c, axis=0)
        # Προσθέτουμε μικρό αριθμό στη διαγώνιο (regularization)
        # για να αποφύγουμε μη αντιστρέψιμο πίνακα
        cov_c = np.cov(X_c, rowvar=False) + np.eye(X_c.shape[1]) * 1e-6

        means.append(mean_c)
        covariances.append(cov_c)
    return np.array(means), np.array(covariances)

def mahalanobis_dist(x, mean, cov_inv):
    diff=x-mean
    return np.sqrt(diff.T @ cov_inv @ diff)

def mahalanobis_classifier(X, means, covariances):
    """
        Ταξινομεί κάθε σημείο στην κλάση με τη μικρότερη απόσταση Mahalanobis.
    """
    n_classes=len(means)
    cov_invs=[np.linalg.inv(cov) for cov in covariances]

    predictions=[]

    for x in X:
        distances=[mahalanobis_dist(x, means[c], cov_invs[c]) for c in range(n_classes)]
        predictions.append(np.argmin(distances))
    return np.array(predictions)

def evaluate(y_true, y_pred, n_classes):
    """
    Accuracy & πίνακας σύγχυσης
    """
    accuracy=np.mean(y_true==y_pred)*100
    print(f"Accuracy: {accuracy:.2f}%")

    cm=np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred]+=1
    return accuracy, cm

def run_third_task(X_all, y_all):
    print("\n---Part 3---")

    # Βήμα 1: PCA σε όλες τις 10 κλάσεις, 10 διαστάσεις
    print("Applying PCA(10 components) in all 10 classes")
    X_pca, W, mean=pca(X_all, components=10)

    # Χωρισμός σε train (1000/κλάση) και test (υπόλοιπα)
    # Χρησιμοποιούμε τα πρώτα 1000 ανά κλάση για εκπαίδευση
    train_idx, test_idx = [], []
    for c in range(10):
        idx = np.where(y_all == c)[0]
        train_idx.extend(idx[:1000])
        test_idx.extend(idx[1000:])

    X_train = X_pca[train_idx]
    y_train = y_all[train_idx]
    X_test = X_pca[test_idx]
    y_test = y_all[test_idx]

    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Βήμα 2: Ταξινομητής με 10 κλάσεις (απλός Mahalanobis)

    print("\n[Classifier 1] Nearest centroid with Mahalanobis (10 classes)")

    # Τα labels εδώ είναι 0-9, οπότε περνάμε y_train απευθείας
    means_10, covs_10 = compute_class_stats(X_train, y_train, n_classes=10)
    y_pred_10 = mahalanobis_classifier(X_test, means_10, covs_10)
    acc_10, cm_10 = evaluate(y_test, y_pred_10, n_classes=10)

    # Βήμα 3: Χωρισμός κάθε κλάσης σε 2 υποκλάσεις με k-means
    print("\n[Classifier 2] Splitting each class into 2 subclusters...")

    # Για κάθε κλάση τρέχουμε k-means με k=2
    # Δημιουργούμε νέα labels: κλάση c → υποκλάσεις 2c και 2c+1
    subclass_labels_train = np.zeros(len(X_train), dtype=int)

    for c in range(10):
        mask = y_train == c
        X_c = X_train[mask]
        local_labels, _ = kmeans(X_c, k=2, random_state=42)
        subclass_labels_train[mask] = 2 * c + local_labels

    # Βήμα 4: Mahalanobis ως προς τις 20 υποκλάσεις
    means_20, covs_20 = compute_class_stats(X_train, subclass_labels_train, n_classes=20)
    y_pred_20_sub = mahalanobis_classifier(X_test, means_20, covs_20)
    y_pred_20 = y_pred_20_sub // 2

    acc_20, cm_20 = evaluate(y_test, y_pred_20, n_classes=10)

    # Οπτικοποίηση confusion matrices
    plot_confusion_matrix(cm_10, 'Confusion Matrix - 10 classes (87.14%)')
    plot_confusion_matrix(cm_20, 'Confusion Matrix - 20 subclasses (88.65%)')

    return acc_10, cm_10, acc_20, cm_20

def plot_confusion_matrix(cm, title):
    """
    Οπτικοποίηση confusion matrix με χρώματα.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    # Αριθμοί μέσα σε κάθε κελί
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=8)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    ticks = list(range(cm.shape[0]))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.tight_layout()
    plt.show()

