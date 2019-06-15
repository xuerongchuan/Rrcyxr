# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from sklearn.model_selection import KFold
from utils.timeHelper import getDate,getTime


def splitByUser(data_path):
    if not os.path.isfile(data_path):
        print('the format of data path is wrong!')
        sys.exit()
    data = pd.read_csv(data_path, header=None, names=['uid', 'iid', 'rating'])
    user_list = list(data.uid.unique())
    cv1, cv2, cv3, cv4, cv5 = [], [], [], [], []

    for user in user_list:
        res = []
        user_data = data[data.uid == user].reset_index(drop=True)
        kf = KFold(5)
        for train_index, test_index in kf.split(user_data):
            res.append(user_data.iloc[test_index])
        cv1.append(res[0])
        cv2.append(res[1])
        cv3.append(res[2])
        cv4.append(res[3])
        cv5.append(res[4])
    pd.concat(cv1).to_csv('data/cv/data-0.csv', header=False, index=False)
    pd.concat(cv2).to_csv('data/cv/data-1.csv', header=False, index=False)
    pd.concat(cv3).to_csv('data/cv/data-2.csv', header=False, index=False)
    pd.concat(cv4).to_csv('data/cv/data-3.csv', header=False, index=False)
    pd.concat(cv5).to_csv('data/cv/data-4.csv', header=False, index=False)


def cut_9data():
    c1 = getTime('2000-9-1')
    c2 = getTime('2001-1-1')
    c3 = getTime('2001-5-1')
    c4 = getTime('2001-9-1')
    c5 = getTime('2002-1-1')
    c6 = getTime('2002-5-1')
    c7 = getTime('2002-9-1')
    c8 = getTime('2003-1-1')
    c9 = getTime('2003-3-1')
    return [0,c1,c2,c3,c4,c5,c6,c7,c8,c9]

def cut_31Data():
    c1 = getTime('2000-9-1')
    c2 = getTime('2000-10-1')
    c3= getTime('2000-11-1')
    c4 = getTime('2000-12-1')
    c5 = getTime('2001-1-1')
    c6 = getTime('2001-2-1')
    c7 = getTime('2001-3-1')
    c8 = getTime('2001-4-1')
    c9 = getTime('2001-5-1')
    c10 = getTime('2001-6-1')
    c11 = getTime('2001-7-1')
    c12 = getTime('2001-8-1')
    c13 = getTime('2001-9-1')
    c14 = getTime('2001-10-1')
    c15 = getTime('2001-11-1')
    c16 = getTime('2001-12-1')
    c17 = getTime('2002-1-1')
    c18 = getTime('2002-2-1')
    c19 = getTime('2002-3-1')
    c20 = getTime('2002-4-1')
    c21 = getTime('2002-5-1')
    c22 = getTime('2002-6-1')
    c23 = getTime('2002-7-1')
    c24 = getTime('2002-8-1')
    c25 = getTime('2002-9-1')
    c26 = getTime('2002-10-1')
    c27 = getTime('2002-11-1')
    c28 = getTime('2002-12-1')
    c29 = getTime('2003-1-1')
    c30 = getTime('2003-2-1')
    c31 = getTime('2003-3-1')
    return [0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19, \
            c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31]
    