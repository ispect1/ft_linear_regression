#!/usr/bin/env python3
from model import MyLinearRegression, Scaler
from config import MODEL_DATA_PATH, SCALER_DATA_PATH
import argparse


if __name__ == "__main__":        
    lin_reg = MyLinearRegression()
    scaler = Scaler()
    INPUT_INFO = 'Various objects for predict. Separate the signs of one object with a comma\n'
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('objects', nargs='+', help=INPUT_INFO)
    args = parser.parse_args()

    filename = MODEL_DATA_PATH
    try:
        lin_reg.download(filename)

        filename = SCALER_DATA_PATH
        scaler.download(filename)
    except FileNotFoundError:
        print(f"Not found file '{filename}'")
    except KeyError:
        print('Not found valid args')
        exit()

    try:
        print(INPUT_INFO)
        X = [list(map(float, obj.split(','))) for obj in args.objects]
    except ValueError:
        print('Enter valid data (int or float)')
        exit()
    X = scaler.transform(X)
    predict = lin_reg.predict(X)
    INPUT_LINE = 'Input'
    LJUST = max(max(map(len, args.objects)), len(INPUT_LINE))
    print(f'{INPUT_LINE.ljust(LJUST)} | Predict',
          *map(lambda x: f'{str(x[0]).ljust(LJUST)} | {str(round(x[1], 3))}',
               zip(args.objects, predict)), sep='\n')
