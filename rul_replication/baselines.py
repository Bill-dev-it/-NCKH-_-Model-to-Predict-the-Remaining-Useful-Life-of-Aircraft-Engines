from __future__ import annotations

import numpy as np


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def train_predict_sklearn(name: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    train_flat = flatten_windows(train_x)
    test_flat = flatten_windows(test_x)
    if name == "sgd":
        model = make_pipeline(
            StandardScaler(),
            SGDRegressor(
                penalty=None,
                alpha=0.0001,
                epsilon=0.1,
                loss="epsilon_insensitive",
                max_iter=5000,
                random_state=42,
            ),
        )
    elif name == "svr":
        model = make_pipeline(StandardScaler(), SVR(kernel="rbf", gamma=0.01, C=1.0, epsilon=0.1))
    else:
        raise ValueError(f"Unknown sklearn model: {name}")
    model.fit(train_flat, train_y)
    return model.predict(test_flat).astype(np.float64)
