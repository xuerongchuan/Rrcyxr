import sys
import os
import json
import numpy as np
import pandas as pd
import time


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
    def init_data(self):
        print('loading...')
        begin = time.time()
        self.preprocess_train_data()
        self.preprocess_test_data()
        end = time.time()
        print('data has been loaded ,used time:{:.4f}s'.format(end-begin))
    def preprocess_train_data(self):
        train_data = pd.read_csv(self.config.train_path)
        self.items = []
        self.months = []
        self.seasons = []
        self.days = []
        self.ratings = []
        for u in range(self.num_users):
            item_u = train_data[train_data.userId == u].movieId.values
            month_u = train_data[train_data.userId == u].month.values
            season_u = train_data[train_data.userId == u].season.values
            day_u = train_data[train_data.userId == u].day.values
            rating_u = train_data[train_data.userId == u].rating.values
            self.items.append(item_u)
            self.months.append(month_u)
            self.seasons.append(season_u)
            self.days.append(day_u)
            self.ratings.append(rating_u)

    def preprocess_test_data(self):
        test_data = pd.read_csv(self.config.test_path)
        self.test_neg= []
        with open('data/ml-1m.test.negative','r') as f:
            for line in f.readlines():
                values = [int(i) for i in line.strip().split('\t')[1:]]
                self.test_neg.append(values)
        self.oitems = list(test_data.movieId.values)
        self.omonths = list(test_data.month.values)
        self.oseasons = list(test_data.season.values)
        self.odays = list(test_data.day.values)
        self.oratings = list(test_data.rating.values)
    
    

class getBatchData(object):
    def __init__(self, config, dl):
        self.dl = dl
        self.config = config
        if self.config.mode == 'season':
            self.numT = 9
            self.times = self.dl.seasons
            self.otimes = self.dl.seasons
        elif self.config.mode == 'month':
            self.numT = 31
            self.times = self.dl.months
            self.otimes = self.dl.omonths
        elif self.config.mode == 'day':
            self.numT = 1039
            self.times = self.dl.days
            self.otimes = self.dl.odays
        else:
            print('错误的mode')
    
    def _generate_neg_items(self, uhist):
        negative_items = []
        for tmp in range(self.config.neg_count):
            j = np.random.choice(self.dl.num_items)
            while j in uhist:
                j = np.random.choice(self.dl.num_items)
            #jt = self.dl.ITmap[str(j)] if str(j) in self.dl.ITmap.keys() else self.dl.numT
            negative_items.append(j)
        return negative_items

    def getTrainBatches(self):
        u_index = list(range(self.dl.num_users))
        np.random.shuffle(u_index)
        for u in u_index:
            uhist = list(self.dl.items[u])
            times = list(self.times[u])
            u_batches = []
            t_batches = []
            i_batches = []
            num_batches = []
            ot_batches = []
            label_batches = []
            for index, (i, t) in enumerate(zip(uhist, times)):
                chist = uhist.copy()
                ctime = times.copy()
                chist.pop(index)
                ctime.pop(index)
                chist = list(chist)+[self.dl.num_items]
                ctime = list(ctime)+[self.numT]
                u_batches.append(chist)
                t_batches.append(ctime)
                i_batches.append(i)
                num_batches.append(len(chist)-1)
                ot_batches.append(t)
                label_batches.append(1)
                neg_items = self._generate_neg_items(uhist)
                for neg_i in neg_items:
                    u_batches.append(uhist)
                    t_batches.append(times)
                    i_batches.append(neg_i)
                    num_batches.append(len(uhist))
                    ot_batches.append(self.numT)
                    label_batches.append(0)
            batches_index = list(range(len(u_batches)))
            np.random.shuffle(batches_index)
            u_batches = [u_batches[i] for i in  batches_index]
            num_batches = [num_batches[i] for i in  batches_index]
            i_batches = [i_batches[i] for i in  batches_index]
            t_batches = [t_batches[i] for i in  batches_index]
            ot_batches = [ot_batches[i] for i in  batches_index]
            label_batches = [label_batches[i] for i in  batches_index]
            yield u_batches, num_batches, i_batches, t_batches, ot_batches, label_batches
    def getTestData(self, uData):
        u_index = list(range(self.dl.num_users))
        np.random.shuffle(u_index)
        for u in u_index:
            uhist = list(self.dl.items[u])
            times = list(self.times[u])
            u_test_neg = self.dl.test_neg[u]

            u_batches = []
            t_batches = []
            i_batches = []
            num_batches = []
            ot_batches = []
            label_batches = [0]*99+[1]
            for neg_i in u_test_neg:
                u_batches.append(uhist)
                t_batches.append(times)
                i_batches.append(neg_i)
                num_batches.append(len(uhist))
                ot_batches.append(self.numT)
            u_batches.append(uhist)
            t_batches.append(times)
            i_batches.append(self.oitems[u])
            num_batches.append(len(uhist))
            ot_batches.append(self.otimes[u])

            batches_index = list(range(len(u_batches)))
            np.random.shuffle(batches_index)
            u_batches = [u_batches[i] for i in  batches_index]
            num_batches = [num_batches[i] for i in  batches_index]
            i_batches = [i_batches[i] for i in  batches_index]
            t_batches = [t_batches[i] for i in  batches_index]
            ot_batches = [ot_batches[i] for i in  batches_index]
            label_batches = [label_batches[i] for i in  batches_index]
            yield u_batches, num_batches, i_batches, t_batches, ot_batches, label_batches
    # def getNormalInputData(self, u, uData):
     
    #     u_batches = []
    #     ut_batches = []
    #     t_batches = []
    #     i_batches = []
    #     label_batches = []
    #     uhist , times = zip(*uData)
    #     uhist = list(uhist)
    #     times = list(times)
    #     tu = int(np.mean(times))
    #     for index, (i, t) in enumerate(uData):
    #         u_batches.append(u)
    #         ut_batches.append([u, t])
    #         i_batches.append(i)
    #         t_batches.append(t)
    #         label_batches.append(1)
    #         neg_items = self._generate_neg_items(uhist)
    #         for neg_i, neg_t in neg_items:
    #             u_batches.append(u)
    #             ut_batches.append([u, neg_t])
    #             i_batches.append(neg_i)
    #             t_batches.append(neg_t)
    #             label_batches.append(0)
    #     return u_batches, ut_batches, i_batches, label_batches,t_batches, tu
    
    # def getNormalTestData(self, u, uData):
    #     oneData = next(self.dl.testset())
    #     u_batches = []
    #     ut_batches = []
    #     t_batches = []
    #     i_batches = []
    #     label_batches = [0]*99+[1]
    #     uhist , times = zip(*uData)
    #     uhist = list(uhist)
    #     times = list(times)
    #     tu = int(np.mean(times))
    #     for it in oneData:
    #         u_batches.append(u)
    #         ut_batches.append([u, it[1]])
    #         t_batches.append(it[1])
    #         i_batches.append(it[0])
    #     return u_batches, ut_batches, i_batches, label_batches,t_batches, tu


            
            
            
                           

            
            
        
        
        