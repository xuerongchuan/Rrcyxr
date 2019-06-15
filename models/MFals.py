# -*- coding: utf-8 -*-

import numpy as np 
import scipy.sparse as sp
#import pandas as pd
from implicit.als import AlternatingLeastSquares
import math

class ALS(object):
    def __init__(self,config, dl):
        self.config = config
        self.dl = dl
        self.train = self.get_trainmat()

#    def build_model(self):
#        print('creating data matrix...')
#        
#        test = sp.lil_matrix(train_data.shape)
#        for u, uData in enumerate(self.dl.testset()):
#            items, _  = zip(*uData)
#            for i in items:
    def get_trainmat(self):
        train = sp.lil_matrix((self.dl.num_users, self.dl.num_items))
        for u,uData in enumerate(self.dl.trainset()):
            items, _  = zip(*uData)
            for i in items:
                train[u,i] = 1
                for tmp in range(self.config.neg_count):
                    j = np.random.choice(self.dl.num_items)
                    while j in items:
                        j = np.random.choice(self.dl.num_items)
                    train[u,j] = 0
        return train.tocsr()
    def evaluate(self):
        hits, ndcgs = [],[]
        for u, uData in enumerate(self.dl.testset()):
            items,_ = zip(*uData)
            items = np.array(items, dtype = np.int64)
            user_factor = self.model.user_factors[u].reshape((1,self.model.factors))
            item_factors = self.model.item_factors[items]
            predictions = np.dot(user_factor, item_factors.transpose()).reshape(-1)
            neg_predict, pos_predict = np.array(predictions[:-1]), np.array(predictions[-1])
            position = (neg_predict >= pos_predict).sum()
            #print(position)
            hr = position < 10
            ndcg = math.log(2) / math.log(position+2) if hr else 0
            hits.append(hr)
            ndcgs.append(ndcg)  
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('hr:{0}, ndcg:{1}'.format(hr, ndcg))
            
    def train_and_evaluate(self):
        self.model = AlternatingLeastSquares(factors= self.config.factors, \
                                        iterations = self.config.epoches)
        self.model.fit(self.train.transpose())
        self.evaluate()
        
                    
        

