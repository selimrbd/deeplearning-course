from pathlib import Path

import joblib


def load_catset():
    """
    Load the train and test set.

    Returns:
        - (X_train, Y_train, X_test, Y_test)
    """

    path = Path("../../data/catvnoncat/data.pkl")
    with open(path, "rb") as f:
        data = joblib.load(f)

    X_train = data["train_set_x"]
    Y_train = data["train_set_y"]
    X_test = data["test_set_x"]
    Y_test = data["test_set_y"]

    return X_train, Y_train, X_test, Y_test


def visualize_catset(X, Y):
    """
    Displays a random image from the CATVNONCAT dataset with its label
    """
    m = X.shape[0]
    i = random.randint(0, m - 1)
    x = X[i]
    y = Y[i]
    labels = {0: "non-cat", 1: "cat"}
    print(f"label: {labels[y]}")
    plt.imshow(x)


def preprocess_catset(X, Y=None):
    """Preprocess input data, in order to be fed as the input of the neural network.

    Operations:
        - Reshaping of the arrays to fit the Neural Network Input
        - Normalization
    Returns:
        (X_pp, Y_pp)

        using the following notations:
        - n_x: number of features in the input of the neural network
        - m: number of observations
        X_pp: normalized matrix of size (n_x, m)
        Y_pp: normalized array of size (1, m)
    """
    m = X.shape[0]

    ## Reshape the arrays
    ### START CODE ###
    X = X.reshape(m, -1).T
    if Y is not None:
        Y = Y.reshape(m, 1).T
    ### END CODE ###

    ## normalization
    ### START CODE ###
    X = X / 255.0
    ### END CODE ###
    return X, Y


def load_and_prepare_catset():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_catset()
    X_train, Y_train = preprocess_catset(X_train_orig, Y_train_orig)
    X_test, Y_test = preprocess_catset(X_test_orig, Y_test_orig)
    return X_train, Y_train, X_test, Y_test
