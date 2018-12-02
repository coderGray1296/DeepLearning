import pandas as pd
import numpy as np
import os
import copy

def choose():
    '''
    techid = 0
    scaleid = 7
    path = '/Users/codergray/PycharmProjects/Energy/data/'
    children = os.listdir(path)
    #获取所有excel源数据的绝对路径
    children = [os.path.join(path,child) for child in children]
    result_data = []

    #选出符合techid和scaleid条件的数据写入excel

    for child in children:
        buf_data = pd.read_excel(child)
        result = buf_data.loc[(buf_data['techid'] == techid) & (buf_data['scaleid'] == scaleid)]
        result_data.append(result)
    for i in range(len(result_data)):
        result_data[i].to_excel('/Users/codergray/PycharmProjects/Energy/data/'+str(i)+'.xls')
    '''
    path = '/Users/codergray/PycharmProjects/Energy/data/0-7.xls'

    #过滤掉存在空值的数据
    data = pd.read_excel(path)

    for i in range(len(data)):
        if data.loc[i]['feedtotal'] == np.nan or data.loc[i]['fueltotal'] == np.nan or data.loc[i]['steamtotal'] == np.nan or pd.isnull(data.loc[i]['watertotal']) or \
                data.loc[i]['electricity'] == np.nan or data.loc[i]['sec'] == np.nan or data.loc[i]['ethylene'] == np.nan or \
                data.loc[i]['propylene'] == np.nan or data.loc[i]['cfour'] == np.nan:
            data.drop([i], inplace=True)
    data.to_excel('/Users/codergray/PycharmProjects/Energy/data/0-7_new.xls')


c = choose()
