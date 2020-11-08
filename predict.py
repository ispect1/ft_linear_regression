#!/usr/bin/env python3
from model import MyLinearRegression, Scaler
from config import MODEL_DATA_PATH, SCALER_DATA_PATH


if __name__ == "__main__":        
    lin_reg = MyLinearRegression()
    scaler = Scaler()

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
        print("INFO:\n\tfeatures - ','\n\tobjects - ' '")
        mileage = [list(map(float, obj.split(','))) for obj in input('Enter data:\t').split(' ')]
        # if sum(True if m[0] < 0 else False for m in mileage):
        #     print('Mileage cannot be less than zero')
        #     exit()
    except ValueError:
        print('Enter valid data (int or float)')
        exit()
    X = scaler.transform(mileage)
    predict = lin_reg.predict(X)
    print('Predict:\t', *map(round, predict))
