from typing import List


class NotFittedError(Exception):
    pass


def mserror(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return sum((i - j)**2 for i, j in zip(y_true, y_pred)) / len(y_true)


def maerror(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return sum(abs(i - j) for i, j in zip(y_true, y_pred)) / len(y_true)


def r2(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    y_mean = mean(y_true)
    return 1 - mserror(y_true, y_pred) * len(y_true) / sum((y_i - y_mean) ** 2 for y_i in y_true)


def dot(X: List[List[float]], v: List[float]):
    assert len(v) == len(X[0])
    return [sum(x_i_j * v_j for x_i_j, v_j in zip(x_i, v)) for x_i in X]


def vectors_diff(vec1, vec2):
    return [v_1 - v_2 for v_1, v_2 in zip(vec1, vec2)]


def mean(arr):
    return sum(arr) / len(arr)


def div(arr):
    arr_mean = mean(arr)
    return 1 / ((len(arr) - 1) or 1) * sum((curr - arr_mean) ** 2 for curr in arr)


def std(arr):
    return div(arr) ** 0.5
