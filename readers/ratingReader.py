import os
import sys
from collections import defaultdict

class RatingGetter(object):
    """获取将第k折数据作为测试集的数据集，并计算保存一些必要的属性"""
    
    def __init__(self, config):
        #一些必要的统计数据信息
        #self.current_k = k #选取第k个cv集作为测试集
        self.user= {} #user2id字典
        self.item= {} #item2id字典
        self.all_user = {} # 数据集中所有的用户字典
        self.all_item = {}
        self.id2user = {}
        self.id2item = {}
        self.dataSet_u = defaultdict(dict)
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self.testSet_i = defaultdict(dict)
        self.testColdUserSet_u = defaultdict(dict)
        self.trainHotUserSet = []
        self.trainSetLength = 0
        self.testSetLength = 0
        self.config = config
        
        # 获取训练集中的平均得分， 为测试集中的冷启动用户或者物品进行推荐
        self.userMeans = {} #保存用户的平均评分
        self.itemMeans = {} #保存商品的平均得分
        self.globalMean = 0 # 数据集的平均评分
        
        #读取数据并获取上述属性值
        self.generate_data_set() 
        self.get_data_statistics()
        
    def trainSet(self):
        data_path = self.config.train_path
        if not os.path.isfile(data_path):
            print('the format of rating data is wrong!', data_path)
            sys.exit()
        with open(data_path, 'r') as f:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                if self.config.test and index == 10000:
                    break
                u, i, r,_ = line.strip('\r\n').split(self.config.sep)
                yield (int(u), int(i), float(r))
    def testSet(self):
        data_path = self.config.train_path
        if not os.path.isfile(data_path):
            print('the format of rating data is wrong', data_path)
            sys.exit()
        with open(data_path) as f:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                if self.config.test and index == 10000:
                    break
                u, i ,r,_ = line.strip('\r\n').split(self.config.sep)
                yield (int(u), int(i), float(r))
    def weighted_sampling(self, data):
        pass
    def get_train_size(self):
        return (len(self.user), len(self.item))
    def generate_data_set(self):
        for index, line in enumerate(self.trainSet()):
            u, i, r = line
            if not u in self.user:
                self.user[u] = len(self.user)
                self.id2user[self.user[u]] = u
            if not i in self.item:
                self.item[i] = len(self.item)
                self.id2item[self.item[i]] = i
            self.trainSet_u[u][i] = r
            self.trainSet_i[i][u] = r
            self.trainSetLength = index+1
        self.all_user.update(self.user)
        self.all_item.update(self.item)
        
        for index, line in enumerate(self.testSet()):
            u, i, r = line
            if not u in self.user:
                self.all_user[u] = len(self.all_user)
            if not i in self.item:
                self.all_item[i] = len(self.all_item)
            self.testSet_u[u][i] = r
            self.testSet_i[i][u] = r
            self.testSetLength = index +1
    def containsUser(self, u):
        if u in self.user:
            return True
        else:
            return False
    def containsItem(self, i):
        if i in self.item:
            return True
        else:
            return False
    def containsUserItem(self, u, i):
        if u in self.trainSet_u:
            if i in self.trainSet_u[u]:
                return True
    def get_data_statistics(self):
        '''针对训练集中用户和商品的统计'''
        total_rating = 0.0
        total_length = 0
        for u in self.user:
            u_total = sum(self.trainSet_u[u].values())
            u_length = len(self.trainSet_u[u])
            total_rating += u_total
            total_length += u_length
            self.userMeans[u] = u_total/float(u_length)
        for i in self.item:
            self.itemMeans[i] = sum(self.trainSet_i[i].values())/ \
            float(len(self.trainSet_i[i]))
        if total_length == 0:
            self.globalMeans =0 
        else:
            self.globalMean = total_rating/total_length
    def get_col(self, c):
        return self.trainSet_i[c]
    def get_row(self, r):
        return self.trainSet_u[r]
            
        