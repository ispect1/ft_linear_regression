import json
import random
from utils import dot, vectors_diff, mean, std


class Saver:
    def __init__(self, **kwargs):
        self.attrs = [key for key in kwargs]

    def save(self, filename):
        json_data = json.dumps({key: getattr(self, key) for key in self.attrs})
        with open(filename, 'w') as f:
            f.write(json_data)
        return self

    def download(self, filename):
        with open(filename) as f:
            data = json.load(f)
        for key in self.attrs:
            setattr(self, key, data[key])
        return self


class MyLinearRegression(Saver):
    def __init__(self):
        self._max_weight_dist = 1e-8
        self._eta = 1e-2
        self._max_iter = 1e4
        self._seed = 21
        self.w = [0, 0]
        super().__init__(w=self.w)

    @property
    def coef_(self):
        return self.w

    def linear_prediction(self, X, w):
        return dot(X, w)

    def predict(self, X):

        X_ = list(map(lambda x: [1, *x], X))

        return self.linear_prediction(X_, self.coef_)

    def stochastic_gradient_step(self, X, y, w, train_ind, eta=0.01):
        x = X[train_ind]
        y = y[train_ind]
        common = sum(x_j * w_j for x_j, w_j in zip(x, w)) - y
        gradient = [eta * x_i * common for x_i in x]
        return vectors_diff(w, gradient)

    def full_gradient_step(self, X, y, w, eta=0.01):
        commons = vectors_diff(dot(X, w), y)
        gradient = [eta * sum(common * x_i[feature_idx] for common, x_i in zip(commons, X)) / len(X)
                    for feature_idx in range(len(X[0]))]
        return vectors_diff(w, gradient)

    def fit(self, X, y, eta=None, max_iter=None, max_weight_dist=None, seed=21, mode='stochastic'):
        assert mode in ('full', 'stochastic')
        if mode == 'full':
            max_iter = max_iter or 1e2
            eta = eta or 1
        else:
            max_iter = max_iter or 1e5
            eta = eta or 0.01
        max_weight_dist = max_weight_dist or 1e-6
        X_ = list(map(lambda x: [1, *x], X))
        # Инициализируем расстояние между векторами весов на соседних
        # итерациях большим числом. 
        weight_dist = float('inf')
        # Инициализируем вектор весов
        w = [0 for _ in range(len(X_[0]))]
        # Счетчик итераций
        iter_num = 0
        # Будем порождать псевдослучайные числа 
        # (номер объекта, который будет менять веса), а для воспроизводимости
        # этой последовательности псевдослучайных чисел используем seed.
        random.seed(seed)

        # Основной цикл
        while weight_dist > max_weight_dist and iter_num < max_iter:
            # порождаем псевдослучайный 
            # индекс объекта обучающей выборки
            iter_num += 1
            if mode == 'stochastic':
                random_ind = random.randint(0, len(X_) - 1)
                w_tmp = self.stochastic_gradient_step(X_, y, w, random_ind, eta=eta)
            else:
                w_tmp = self.full_gradient_step(X_, y, w, eta=eta)

            w = w_tmp

        self.w = w

        return self


class Scaler(Saver):
    def __init__(self):
        self.std = None
        self.mean = None
        self.count_feature = None
        self._is_trained = False
        super().__init__(_is_trained=self._is_trained, mean=self.mean, std=self.std,
                         count_feature=self.count_feature)

    def fit(self, X):
        self._is_trained = True

        self.std = []
        self.mean = []
        self.count_feature = len(X[0])
        for feature_idx in range(self.count_feature):
            data = [x[feature_idx] for x in X]
            curr_mean = mean(data)
            curr_std = std(data)
            self.mean.append(curr_mean)
            self.std.append(curr_std)

        return self

    def transform(self, X):
        if not self._is_trained:
            return X
        assert len(X[0]) == self.count_feature, f'X has {len(X[0])} features, but this scaler is expecting' \
                                                f' {self.count_feature} features as input.'
        return [[(x - curr_mean) / (curr_std or 1)
                 for x, curr_mean, curr_std in zip(x_i, self.mean, self.std)] for x_i in X]

    def fit_transform(self, X):
        return self.fit(X).transform(X)
