# -*- coding: utf-8 -*-
class Config(object):
    """保存可能需要的参数，并为其设置默认值"""
    
    def __init__(self):
        

        self.verbose = True  #是否输出预测过程
        self.verbose_count = 2
        self.epoches = 1000
        self.learning_rate= 0.01

class mfConfig(Config):
    def __init__(self):
        super(mfConfig, self).__init__()
        self.size = 0.8 #训练集大小
        self.factor = 10 #矩阵分解潜因子维度
        self.lr = 0.01 #学习速率
        self.maxIter = 100 #最大迭代次数
        self.k_fold_num = 5 #交叉验证折数
        self.sep = ','
        self.rating_cv_path = None
        self.dataset_name = None
        self.coldUserRating = 1 #评分数小于这个值的用户被判断为冷启动用户
        self.n = 10 #CF算法的近邻数
        
class naisConfig(Config):
    def __init__(self, mode):
        super(naisConfig, self).__init__()       
        #naisData
        if mode == 'season':
            self.numT = 9
            self.data_path = 'data/season/'
        elif mode == 'month':
            self.numT = 31
            self.data_path = 'data/month/'
        elif mode == 'day':
            self.numT = 1039
            self.data_path = 'data/day/'
        elif mode == 'rating':
            self.data_path = 'data/rating/'
            self.numT = 31
        self.numrows = 10000
        
        self.train_path = self.data_path+'hist_u'
        self.test_path = self.data_path+'test_u'
        self.itmap = self.data_path+'itmap'
        #nais
        
        self.neg_count = 4
        self.time_embedding_size = 16
        self.embedding_size =16
        self.weight_size = 16
        
        self.beta = 0.5
        self.alpha = 0
        regs = [1e-5, 1e-7, 1e-4]
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.eta_bilinear = regs[2]
        self.lr = 0.01
        self.K = 10
        self.len_history = 500
        

class svdConfig(Config):
    def __init__(self):
        super(svdConfig, self).__init__()
        #bpr
        self.k = 16
        self.regU = 0.01
        self.regI = 0.01
        #svd
        self.regB1= 1e-5
        self.regB2 = 1e-5
        self.regU3 = 1e-5
        self.factors = 16
        self.batch_size = 256
        self.neg_count = 4