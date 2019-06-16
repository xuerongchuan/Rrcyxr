


import sys
import os
import json
import numpy as np



class Dataloader(object):
    '''
    导入nais的数据
    按照用户作为batches
    数据要求：
    1.用户商品id都经过映射从0开始
    '''
    def __init__(self, config):
        self.config = config
        self.num_items = 3706 #数据预处理时计算的
        self.num_users = 6040
        self.numT = self.config.numT
        
     
    def trainset(self, numrow=10):
        if not os.path.isfile(self.config.train_path):
            print('输入路径有问题', self.config.train_path)
            sys.exit()
        with open(self.config.train_path, 'r') as f:
            i=0
            for line in f:
                data = json.loads(line)
                if len(data) > self.config.len_history:
                    data = data[:self.config.len_history]
                if i>numrow:
                    break
                i+=1
                yield data
                
    
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
    
            
            


class getBatchData(object):
    def __init__(self, config, dl):
        self.dl = dl
        self.config = config
    

    def getInputData(self, uData):
        u_batches = []
        t_batches = []
        i_batches = []
        num_batches = []
        ot_batches = []
        label_batches = []
        uhist , times, ratings = zip(*uData)
        uhist = list(uhist)
        times = list(times)
        for index, (i, t, r) in enumerate(uData):
            chist = uhist.copy()
            ctime = times.copy()
            chist.pop(index)
            ctime.pop(index)
            chist = list(chist)+[self.dl.num_items]
            ctime = list(ctime)+[self.dl.numT]
            u_batches.append(chist)
            t_batches.append(ctime)
            i_batches.append(i)
            num_batches.append(len(uData))
            ot_batches.append(t)
            label_batches.append(r)
        return u_batches, num_batches, i_batches, t_batches, ot_batches, label_batches
    
    def getNormalInputData(self, u, uData):
     
        u_batches = []
        ut_batches = []
        t_batches = []
        i_batches = []
        label_batches = []
        uhist , times = zip(*uData)
        uhist = list(uhist)
        times = list(times)
        tu = int(np.mean(times))
        for index, (i, t) in enumerate(uData):
            u_batches.append(u)
            ut_batches.append([u, t])
            i_batches.append(i)
            t_batches.append(t)
            label_batches.append(1)
            neg_items = self._generate_neg_items(uhist)
            for neg_i, neg_t in neg_items:
                u_batches.append(u)
                ut_batches.append([u, neg_t])
                i_batches.append(neg_i)
                t_batches.append(neg_t)
                label_batches.append(0)
        return u_batches, ut_batches, i_batches, label_batches,t_batches, tu
    def getTestData(self, uData):
        oneData = next(self.dl.testset())
        
        u_batches = []
        t_batches = []
        i_batches = []
        num_batches = []
        ot_batches = []
        label_batches = []
        uhist , times, ratings = zip(*uData)
        uhist = list(uhist)
        times = list(times)
        for it in oneData:
            u_batches.append(uhist)
            t_batches.append(times)
            i_batches.append(it[0])
            num_batches.append(len(uData))
            ot_batches.append(it[1])
            label_batches.append(it[2])
        return u_batches, num_batches, i_batches, t_batches, ot_batches, label_batches
    def getNormalTestData(self, u, uData):
        oneData = next(self.dl.testset())
        u_batches = []
        ut_batches = []
        t_batches = []
        i_batches = []
        label_batches = [0]*99+[1]
        uhist , times = zip(*uData)
        uhist = list(uhist)
        times = list(times)
        tu = int(np.mean(times))
        for it in oneData:
            u_batches.append(u)
            ut_batches.append([u, it[1]])
            t_batches.append(it[1])
            i_batches.append(it[0])
        return u_batches, ut_batches, i_batches, label_batches,t_batches, tu
    def generateTrainData(self):
        for uData in self.dl.trainset(self.config.numrows):
            value = self.getInputData(uData)
            yield value
    def generateTestData(self):
        for uData in self.dl.trainset(self.config.numrows):
            value = self.getTestData(uData)
            yield value
    def generateNormalTrainData(self):
        for u, uData in enumerate(self.dl.trainset(self.config.numrows)):
            value = self.getNormalInputData(u,uData)
            yield value
    def generateNormalTestData(self):
        for u, uData in enumerate(self.dl.trainset(self.config.numrows)):
            value = self.getNormalTestData(u, uData)
            yield value 

            
            
            
                           

            
            
        
        
        