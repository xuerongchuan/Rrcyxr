# -*- coding: utf-8 -*-

import math
import numpy as np
#from collections import defaultdict
from utils.mathfunctions import sigmoid

class BPR(object):
    '''隐式行为算法'''
    def __init__(self, config, dl):
        self.config = config
        self.dl = dl
        self.lRate = self.config.lr
        self.regU = self.config.regU
        self.regI = self.config.regI
        self.initModel()
    def initModel(self):
        self.P = np.random.rand(self.dl.num_users, self.config.k)/3
        self.Q = np.random.rand(self.dl.num_items, self.config.k)/3
    
    def buildModel(self):

        for u, uhist in enumerate(self.dl.trainset(self.config.numrows)):
            history,_ = zip(*uhist)
            for i, t in uhist:
                j = np.random.choice(self.dl.num_items)
                while j in history:
                    j =  np.random.choice(self.dl.num_items)
                self._optimize(u,i,j)
            
            
    def _optimize(self,u,i,j):
        s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]

        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]
        self.loss += -math.log(s)
    
    def train_and_evaluate(self):
        print('training...')
        iter = 0
        while iter < self.config.maxIter:
            self.loss = 0
            self.buildModel()
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            
            if iter % self.config.verbose_count==0 and self.config.verbose:
                hr,ndcg = self.evaluate()
                print('epoch:{0}, traingloss:{3},hr:{1}, ndcg:  {2}'.format(iter,hr,ndcg,self.loss))
            iter += 1 
#            if self.isConverged(iter):
#                break
    def predict(self, u, i):
        yui = sigmoid(self.Q[i].dot(self.P[u]))
        return yui
    def evaluate(self):
        hits, ndcgs = [],[]
        for u , objects in enumerate(self.dl.testset(self.config.numrows)):
            oitems,_ = zip(*objects)
            predictions = [self.predict(u,i) for i in oitems]
            neg_predict, pos_predict = np.array(predictions[:-1]), np.array(predictions[-1])
            position = (neg_predict >= pos_predict).sum()
            #print(position)
            hr = position < 10
            ndcg = math.log(2) / math.log(position+2) if hr else 0
            hits.append(hr)
            ndcgs.append(ndcg)  
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr,ndcg

            
        
    