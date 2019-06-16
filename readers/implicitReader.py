# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

class Dataloader(object):
    
    def __init__(self, config):
        
        self.config = config
        self.uhist = {}
        self.init_data()
        self.batch_size = config.batch_size
    
    def init_data(self):
        train_data = pd.read_csv('data/ml-1m.train.rating', header=None,  \
                                 names=['userId', 'movieId', 'rating', 'timestamp'],sep='\t')
        for u in train_data['userId'].unique():
            self.uhist[u] = list(train_data[train_data['userId']==u].movieId)
        train_data= shuffle(train_data)
        self.trainset = list(zip(train_data['userId'], train_data['movieId']))
        self.train_len = len(train_data)
        self.num_items = 3706
        self.num_users = 6040
    
    def _generate_neg_items(self, uhist):
        negative_items = []
        for tmp in range(self.config.neg_count):
            j = np.random.choice(self.num_items)
            while j in uhist:
                j = np.random.choice(self.num_items)
#            jt = self.dl.ITmap[str(j)]
            negative_items.append(j)
        return negative_items
    
    def batch_gen(self):
        u_batches = []
        i_batches = []
        label_batches = []
        batch_num = self.train_len // self.batch_size
        for batch_i in range(batch_num):
            start = batch_i * self.batch_size
            end = (batch_i+1) * self.batch_size
            for index in  range(start, end):
                u, i = self.trainset[index]
                u_batches.append(u)
                i_batches.append(i)
                label_batches.append(1)
                neg_items = self._generate_neg_items(self.uhist[u])
                for neg_i in neg_items:
                    u_batches.append(u)
                    i_batches.append(neg_i)
                    label_batches.append(0)
            yield u_batches, i_batches, label_batches
    def testset(self, numrow=10):
        if not os.path.isfile(self.config.test_path):
            print('输入路径有问题', self.config.test_path)
            sys.exit()
        with open(self.config.test_path, 'r') as f:
            i=0
            for line in f:
                if i>numrow:
                    break
                i+=1
                yield json.loads(line)            

    def getTestData(self, u):
        oneData = next(self.dl.testset())
        
        u_batches = []
        i_batches = []
        label_batches = [0]*99+[1]

        for it in oneData:
            u_batches.append(u)
            i_batches.append(it[0])
        return u_batches,i_batches,label_batches

    def generateTestData(self):
        for u in range(self.num_users):
            value = self.getTestData(u)
            yield value           
                
            
        
            
        
    