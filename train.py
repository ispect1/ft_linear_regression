#!/usr/bin/env python3
from model import MyLinearRegression, Scaler
import csv
from utils import mserror, maerror, r2
from config import DATA_PATH, MODEL_DATA_PATH, NUM_TARGET_COLUMN, SCALER_DATA_PATH, MAX_ITER, ETA
import argparse


def prepare_data(filename, num_targets_column):
    with open(filename) as f:
        X = []
        y = []
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if len(line) != len(header):
                print('Invalid csv file')
                exit()
            try:
                y.append(float(line.pop(int(num_targets_column))))
            except IndexError:
                print('Invalid target column number')
                exit()
            X.append(list(map(float, list(line))))
    return X, y


def plot_data(X, y, model, scaler):
    if len(X[0]) > 1:
        print('Графики возможны только для двумерного случая')
        return
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print('Matplotlib не установлен. Установите для построения графиков')
        return
    arr = [x[0] for x in X]
    plt.scatter(arr, y, edgecolors='white')
    new_w_0 = model.predict(scaler.transform([[0]]))[0]
    new_w_1 = (model.predict(scaler.transform([[min(arr)]]))[0] - new_w_0) / min(arr)
    line = f'y = {round(new_w_1, 3)} * x + {round(new_w_0, 3)}'
    print('Linear regression formula: ', line)
    plt.plot(X, model.predict(scaler.transform(X)), color='blue', label=line)
    plt.legend()
    plt.show()


def main(args):

    lin_reg = MyLinearRegression()
    scaler = Scaler()
    X, y = prepare_data(args.filename_data, args.num_target_column)
    X_ = scaler.fit_transform(X)
    lin_reg.fit(X_, y, mode=args.gradient_mode, max_iter=args.max_iter, eta=args.eta)
    predict = lin_reg.predict(X_)

    lin_reg.save(MODEL_DATA_PATH)
    scaler.save(SCALER_DATA_PATH)
    if args.metric:
        print(f'RMSE: {mserror(y, predict)**0.5}')
        print(f'MAE: {maerror(y, predict)}')
        print(f'R2: {r2(y, predict)}')

    if args.plot:
        plot_data(X, y, lin_reg, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train  model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plot', '-p', dest='plot', action='store_true', help='plot mode')
    parser.add_argument('--metrics', '-m', dest='metric', action='store_true', help='calculate metrics')
    parser.add_argument('--filename_data', '-f', dest='filename_data', action='store', help='input data file',
                        default=DATA_PATH)
    parser.add_argument('--num_target_column', '-n', dest='num_target_column', action='store',
                        help='target column number', default=NUM_TARGET_COLUMN)
    parser.add_argument('--gradient_mode', '-g', dest='gradient_mode', action='store', help='gradient mode',
                        choices={'full', 'stochastic'}, default='stochastic')
    parser.add_argument('--max_iter', '-i', dest='max_iter', action='store', help='max iter steps', type=int,
                        default=MAX_ITER)
    parser.add_argument('--eta', '-e', dest='eta', action='store', help='eta', type=float, default=ETA)

    arguments = parser.parse_args()
    main(arguments)
