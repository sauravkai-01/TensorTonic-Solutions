import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    train = np.asarray(X_train, dtype=float)
    labels = np.asarray(y_train)
    test = np.asarray(X_test, dtype=float)
    classes, counts = np.unique(labels, return_counts=True)
    priors = counts / labels.size
    probabilities = np.vstack([
        (train[labels == label].sum(axis=0) + 1) / (count + 2)
        for label, count in zip(classes, counts)
    ])
    log_present = test @ np.log(probabilities).T
    log_absent = (1 - test) @ np.log1p(-probabilities).T
    return np.round(np.log(priors) + log_present + log_absent, 4)