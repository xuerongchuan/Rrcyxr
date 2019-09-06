# -*- coding: utf-8 -*-

import sys
sys.path.append("..")
from readers.ratingReader import RatingGetter
from metrics.metric import Metric
import numpy as np
import time

class MF(object):
    '''矩阵分解算法的父类'''
    def __init__(self, config):
        self.config = config
        self.rg = RatingGetter(config)
        pass
    def init_model(self):
        
        self.P= np.random.rand(self.rg.get_train_size()[0], self.config.factor)/ \
        (self.config.factor**0.5)
        self.Q = np.random.rand(self.rg.get_train_size()[1], self.config.factor)/ \
        (self.config.factor**0.5)
        self.loss , self.lastloss = 0.0, 0.0
        self.lastRmse, self.lastMae = 10.0, 10.0
    
    def train_model(self, k):
        self.init_model(k)
        pass
    def predict(self, u ,i ):
        print('wrong')
        if self.rg.containsUser(u) and self.rg.containsItem(i):
            return self.P[self.user[u]].dot(self.Q[self.item[i]])
        elif self.rg.containsUser(u) and not self.rg.containsItem(i):
            return self.rg.userMeans[u]
        elif not self.rg.containsUser(u) and self.rg.containsItem(i):
            return self.rg.itemMeans[i]
        else:
            return self.rg.globalMean
    def predict_model(self):
        '''为测试集中的用户预测'''
        res = []
        for ind, entry in enumerate(self.rg.testSet()):
            user, item, rating = entry
            rating_length = len(self.rg.trainSet_u[user])
            #冷启动用户不进行预测评分
            if rating_length <= self.config.coldUserRating:
                continue
            to = time.time()
            prediction = self.predict(user, item)
            ti = time.time()
            pre_time = to-ti
            if self.config.verbose:
                print(user, item, rating, prediction, pre_time)
            res.append([user, item, rating, prediction])
        rmse = Metric.RMSE(res)
        mae = Metric.MAE(res)
        return rmse, mae
            
            