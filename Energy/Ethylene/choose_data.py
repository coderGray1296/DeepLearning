import pandas as pd
import numpy as np
import os
import copy

def choose():
    techid = 0
    scaleid = 7
    path = '/Users/codergray/PycharmProjects/Energy/data/'
    children = os.listdir(path)
    #获取所有excel源数据的绝对路径
    children = [os.path.join(path,child) for child in children]
    result_data = []

    #选出符合techid和scaleid条件的数据写入excel
    '''
    for child in children:
        buf_data = pd.read_excel(child)
        result = buf_data.loc[(buf_data['techid'] == techid) & (buf_data['scaleid'] == scaleid)]
        result_data.append(result)
    for i in range(len(result_data)):
        result_data[i].to_excel('/Users/codergray/PycharmProjects/Energy/data/'+str(i)+'.xls')
    '''

    #过滤掉存在空值的数据

        for i in range(len(buf_data)):
            
            if buf_data.loc[i]['techid'] == techid and buf_data.loc[i]['scaleid'] == scaleid:
                temp = []
                temp.append(buf_data.loc[i]['feedtotal'])
                temp.append(buf_data.loc[i]['fueltotal'])
                temp.append(buf_data.loc[i]['steamtotal'])
                temp.append(buf_data.loc[i]['watertotal'])
                temp.append(buf_data.loc[i]['electricity'])
                temp.append(buf_data.loc[i]['sec'])

                temp.append(buf_data.loc[i]['ethylene'])
                temp.append(buf_data.loc[i]['propylene'])
                temp.append(buf_data.loc[i]['cfour'])
                result_data.append( copy.deepcopy(temp))



c = choose()