from src.data_loader import load_mnist, select_classes

X,y=load_mnist()

#Επιλογή 3 κλάσεων για το μέρος Α και Β
classes_to_use=[0,1,2]
X_3, y_3=select_classes(X,y,classes=classes_to_use, samples_per_class=1000)

# Μέρος Α - PCA με 2 διαστάσεις
from src.pca_impl import run_pca
X_pca, W, mean = run_pca(X_3, y_3, classes=classes_to_use, components=2)

# Μέρος Β - K-means clustering
from src.k_means import run_kmeans
kmeans_labels, centroids_pca, centroids_original = run_kmeans(
    X_pca, y_3, classes=classes_to_use, W=W, mean=mean, k=3
)

#Μέρος Γ
from src.third_task import run_third_task
acc_10, cm_10, acc_20, cm_20=run_third_task(X, y)