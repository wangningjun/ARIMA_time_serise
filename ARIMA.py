import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA
import os


'''
更新日期：2018/8/15
说明：本脚本的内容是以ARIMA算法预测时间序列
准确预测未来n时刻的数据
'''

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def Arima():
    differenced = difference(X, days_in_day)
    model = ARIMA(differenced, order=(1, 0, 2))
    ARIMA_model = model.fit(disp=0)
    return ARIMA_model

def predict(model):
    # 在训练的时间段内，predict和fittedvalues的效果一样，不同的是predict可以往后预测
    differenced = difference(X, days_in_day)
    start_index = len(differenced)
    end_index = len(differenced) + predict_long
    forecast = model.predict(start=start_index, end=end_index)
    history = [x[0] for x in X]
    # plt.plot(history, color='red', label='predict_data')

    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_day)
        history.append(inverted)
    return history

def show(history):
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(history, color='red', label='predict1')
    plt.subplot(2, 1, 2)
    plt.plot(history, color='red', label='predict2')
    plt.show()

if __name__ == '__main__':
    days_in_day = 24
    predict_long = 23
    path_in = 'data/in/outdata/'  #文件夹路径
    path_out = 'data/out/'
    for file in os.listdir(path_in):
        df = open(path_in+file)  # file
        data = pd.read_csv(df, usecols = [0])
        if len(data)<predict_long:
            continue
        X = data.values
		mode = np.mean(X)
        model= Arima()
        results_pre= predict(model)
        rng = pd.date_range(data.idxmin()[0], periods=len(data)+predict_long+1, freq='H')
        results_pre = pd.Series(results_pre,index=rng)
        path = os.path.join(path_out+file)
        results_pre.to_csv(str(path))
        # show(results_pre)

