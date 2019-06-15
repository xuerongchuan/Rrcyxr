# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from models.mf import MF
from utils.matrix import SimMatrix
from utils.similarity import pearson_sp

class UserCF(MF):
    def __init__(self, config):
        super(UserCF, self).__init__(config)
        #为算法新添加一个参数n，近邻数
        
    def init_model(self, k):
        super(UserCF, self).init_model(k)
        self.user_sim = SimMatrix()
        
        for u_test in self.rg.testSet_u:
            for u_train in self.rg.user:
                if u_test != u_train:
                    if self.user_sim.contains(u_test, u_train):
                        continue
                    sim = pearson_sp(self.rg.get_row(u_test),  \
                                     self.rg.get_row(u_train))
                    self.user_sim.set(u_test, u_train, sim)
    def predict(self, u, i):
        '''基于物品的协同过滤，算法主要的工作都在预测上，这里通过建立物品相似度
        对称矩阵，找到目标商品的所有相似商品列表，然后找到这些商品中目标用户评过分
        的商品计算预测值'''
        matchUsers = sorted(self.user_sim[u].items(), key=lambda x:x[1], \
                            reverse=True)
        userCount = self.config.n
        if userCount > len(matchUsers):
            userCount = len(matchUsers)
        sum, denom = 0, 0
        for n in range(userCount):
            similarUser = matchUsers[n][0]
            if self.rg.containsUserItem(similarUser , i):
                similarity = matchUsers[n][1]
                rating = self.rg.trainSet_u[similarUser][i]
                sum += similarity * (rating-self.rg.userMeans[similarUser])
                denom += similarity
        if sum == 0:
            if  self.rg.containsUser(u):
                return self.rg.userMeans[u]
            return self.rg.globalMean
            
        pred = self.rg.userMeans[u] + sum/float(denom)
        print('啦啦啦啦',pred)
        return pred
        