import sys
sys.path.append('..')
from models.mf import MF
from utils.matrix import SimMatrix
from utils.similarity import pearson_sp

class ItemCF(MF):
    def __init__(self, config):
        super(ItemCF, self).__init__(config)
        #为算法新添加一个参数n，近邻数
        
    def init_model(self, k):
        super(ItemCF, self).init_model(k)
        self.item_sim = SimMatrix()
        
        for i_test in self.rg.testSet_i:
            for i_train in self.rg.item:
                if i_test != i_train:
                    if self.item_sim.contains(i_test, i_train):
                        continue
                    sim = pearson_sp(self.rg.get_col(i_test),  \
                                     self.rg.get_col(i_train))
                    self.item_sim.set(i_test, i_train, sim)
    def predict(self, u, i):
        '''基于物品的协同过滤，算法主要的工作都在预测上，这里通过建立物品相似度
        对称矩阵，找到目标商品的所有相似商品列表，然后找到这些商品中目标用户评过分
        的商品计算预测值'''
        matchItems = sorted(self.item_sim[i].items(), key=lambda x:x[1], \
                            reverse=True)
        itemCount = self.config.n
        if itemCount > len(matchItems):
            itemCount = len(matchItems)
        sum, denom = 0, 0
        for n in range(itemCount):
            similarItem = matchItems[n][0]
            if self.rg.containsUserItem(u , similarItem):
                similarity = matchItems[n][1]
                rating = self.rg.trainSet_u[u][similarItem]
                sum += similarity * (rating-self.rg.itemMeans[similarItem])
                denom += similarity
        if sum == 0:
            if  self.rg.containsItem(i):
                return self.rg.itemMeans[i]
            return self.rg.globalMean
            
        pred = self.rg.itemMeans[i] + sum/float(denom)
        return pred
        