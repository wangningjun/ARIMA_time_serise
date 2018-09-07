import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA
import os
from pandas.core.frame import DataFrame
from sys import maxsize

'''
更新日期：2018/8/28
说明：本脚本的内容是以ARIMA算法预测时间序列
更新说明：实现动态调参
准确预测未来n时刻的数据,统计smape
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

def Arima_up(x):
    differenced = difference(x, days_up)
    _, p, q, _ = proper_model(differenced,3)
    print(p,q)
    model = ARIMA(differenced, order=(p, 0, q))
    ARIMA_model = model.fit(disp=0)
    return ARIMA_model

def Arima_down(x):
    differenced = difference(x, days_down)
    _,p,q,_ = proper_model(differenced,3)
    print(p,q)
    model = ARIMA(differenced, order=(p, 0, q))
    ARIMA_model = model.fit(disp=0)
    return ARIMA_model

def predict(X,model,days):
    # 在训练的时间段内，predict和fittedvalues的效果一样，不同的是predict可以往后预测
    differenced = difference(X, days)
    start_index = len(differenced)
    end_index = len(differenced) + predict_long
    forecast = model.predict(start=int(start_index), end=int(end_index))
    history = [x for x in X]
    # plt.plot(history, color='red', label='predict_data')

    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days)
        if inverted<0:    # 预测出来小于0的值全都填为0
            inverted = 0

        history.append(inverted)
    return history

def show(predict,realy):
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(predict, color='red', label='predict')
    plt.subplot(2, 1, 2)
    plt.plot(realy, color='red', label='realy')
    plt.show()

def show_double(predict,realy):
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(predict, color='red', label='predict')
    plt.subplot(2, 1, 2)
    plt.plot(realy, color='red', label='realy')
    plt.show()

def proper_model(data_ts, maxLag):
    init_bic = maxsize
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARIMA(data_ts, order=(p,0,q))
            try:
                results_ARIMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARIMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARIMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel

if __name__ == '__main__':
    days_up = 24
    days_down = 24
    predict_long = 29
    path_in = 'data/in/outdata/'  # 文件夹路径
    path_out = 'data/out/'
    count = 0
    for file in os.listdir(path_in):
        df = open(path_in + file, 'rb')  # file

        data = pd.read_csv(df, header=None)
        data.columns = ['time', 'up', 'down']
        len_data = len(data)
        data_ = data.head(len_data-(predict_long+1))   # 除去最后三十行 都要
        if len(data_) < predict_long:
            continue
        X1 = data_['up']
        X2 = data_['down']
        try:
            model_up = Arima_up(X1)
        except:
            continue
        # try:
        #     model_down = Arima_down(X2)
        # except:
        #     continue
        results_pre_up = predict(X1, model_up, days_up)
        # results_pre_down = predict(X2, model_down, days_down)
        count += 1
        real_up = data['up'].tail(predict_long+1)
        real_down = data['down'].tail(predict_long+1)
        show(results_pre_up, real_up)

        pre_up_long = np.array(results_pre_up[-(predict_long+1):])
        smape_up = sum(abs( pre_up_long- real_up.values)/(pre_up_long+real_up.values))/(predict_long+1)*2
        # smape_down = sum(abs(results_pre_down-real_down.values)/(results_pre_down+real_down.values))/len(results_pre_down)*2

        print(smape_up)

    # print('total:', count)