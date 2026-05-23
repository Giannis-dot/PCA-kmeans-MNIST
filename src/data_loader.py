from datasets import load_dataset
import numpy as np

#LOAD DATA
def load_mnist():
    ds = load_dataset("ylecun/mnist")
    train_data = ds['train']
    X = np.array([np.array(img).flatten() for img in train_data['image']])
    y = np.array(train_data['label'])
    X=X/255.0
    print(f"Loaded Mnist dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y

def select_classes(X,y,classes,samples_per_class=1000):
    """
        Επιλέγει συγκεκριμένες κλάσεις και N δείγματα ανά κλάση.

        classes: λίστα με τα ψηφία που θέλουμε, π.χ. [0, 1, 2]
        samples_per_class: πόσα δείγματα ανά κλάση (default: 1000)
    """
    X_out, Y_out=[], []
    for c in classes:
        indices=np.where(y==c)[0]
        indices=indices[:samples_per_class] # Επιλέγουμε τα πρώτα N δείγματα
        X_out.append(X[indices])
        Y_out.append(y[indices])
    X_out=np.vstack(X_out)
    Y_out=np.concatenate(Y_out)

    print(f"Selected classes {classes} with {samples_per_class} samples each: {X_out.shape[0]} total samples.")
    return X_out, Y_out


