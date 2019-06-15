# -*- coding: utf-8 -*-

import time
import datetime
import time
import math
import re
import pandas as pd
import numpy as np
import scipy.sparse as sp





def getTime(timeVal):
  t = datetime.datetime.strptime(timeVal, '%Y-%m-%d')
  d = t.timetuple()
  return int(time.mktime(d))

def getDate(timestamp):
  timeArray = time.localtime(timestamp)
  timestr = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
  date = datetime.datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
  return date
def generate_item_time_map_dict(data):
    
  grouped = data.groupby(['movieId']).mean()
  return dict(zip(grouped.index, pd.cut(grouped['timestamp'].values, [0,c1,c2,c3,c4,c5,c6,c7,c8,c9], labels=[0,1,2,3,4,5,6,7,8])))
def generate_train_list(data):
    train_list = []
    for u in data.userId.unique():
        train_list.append([(i[0], i[1], i[2])for i in data[data.userId==u][['movieId','label','timestamp']].values])
    return train_list

def get_list():
    numT = 9
    c1 =  getTime('2000-9-1')
    c2 = getTime('2001-1-1')
    c3 = getTime('2001-5-1')
    c4 = getTime('2001-9-1')
    c5 = getTime('2002-1-1')
    c6 = getTime('2002-5-1')
    c7 = getTime('2002-9-1')
    c8 = getTime('2003-1-1')
    c9 = getTime('2003-3-1')
    train_data = pd.read_csv('', header=None, \
                             names=['userId', 'movieId', 'rating', 'timestamp'],sep='\t')
    test_data = pd.read_csv('data/ml-1m.test.rating', header=None, \
                             names=['userId', 'movieId', 'rating', 'timestamp'],sep='\t')
    
    train_data['label'] = np.array([1]*len(train_data))
#    num_items = max(train_data.movieId.unique())+1
#    
#    ITmap = dict(generate_item_time_map_dict(train_data[['movieId','timestamp']]))
    train_data['timestamp']= pd.cut(train_data.timestamp, [0,c1,c2,c3,c4,c5,c6,c7,c8,c9], \
              labels=[0,1,2,3,4,5,6,7,8])
    test_data['timestamp']= pd.cut(test_data.timestamp, [0,c1,c2,c3,c4,c5,c6,c7,c8,c9], \
             labels=[0,1,2,3,4,5,6,7,8])
    
    train_list = generate_train_list(train_data)
    test_data['label'] = np.array([0]*len(test_data))
    test_list = generate_train_list(test_data)
    return train_list, test_list

  