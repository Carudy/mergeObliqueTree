import sklearn.datasets as skd
from sklearn.preprocessing import LabelEncoder

from config import DATA_DIR


def read_dataset(name):
    if name == "wine":
        data = skd.load_wine(return_X_y=True)
    elif name == "digits":
        data = skd.load_digits(return_X_y=True)
    elif name == "iris":
        data = skd.load_iris(return_X_y=True)
    elif name == "breast_cancer":
        data = skd.load_breast_cancer(return_X_y=True)
    else:
        try:
            data = read_libsvm_data(name)
        except FileNotFoundError:
            raise ValueError(f"Dataset '{name}' not found in LIBSVM format.")

    return data


def read_libsvm_data(name):
    from sklearn.datasets import load_svmlight_file

    path = DATA_DIR / name
    X, y = load_svmlight_file(path)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X.toarray(), y
