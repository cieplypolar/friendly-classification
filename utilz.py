import numpy as np
import time
from classification import BaseModel

def split_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = data[:, :-1]
    y = data[:, -1].reshape((-1, 1))
    return X, y

def divide_dataset(
        data: np.ndarray,
        fractions_train_val_test: list[float]
    ) ->tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert len(fractions_train_val_test) == 3
    train_ratio = fractions_train_val_test[0]
    val_ratio = fractions_train_val_test[1]
    # test_ratio - the rest

    classes = np.unique(data[:, -1])
    train, val, test = [], [], []

    for c in classes:
        class_data = data[data[:, -1] == c]
        np.random.shuffle(class_data)
        n_class = len(class_data)
        n_train = int(n_class * train_ratio)
        n_val = int(n_class * val_ratio)

        train.append(class_data[:n_train])
        val.append(class_data[n_train : n_train + n_val])
        test.append(class_data[n_train + n_val :])

    train = np.vstack(train)
    val = np.vstack(val)
    test = np.vstack(test)

    np.random.shuffle(train)
    # latter 2 lines are not necessary
    np.random.shuffle(val)
    np.random.shuffle(test)

    return train, val, test