# -*- coding: utf-8 -*-
import numpy as np
class ktensor(object):
    def __init__(self, U, lmbda=None):
        '''三维的张量，第二维可以不相同，但是第三维必须要相同
        example：
        U = [np.random.rand(i,4) for i in (20, 10, 14)]
        
        '''
        self.U = U
        self.shape = tuple(Ui.shape[0] for Ui in U)
        self.ndim = len(self.shape)
        self.rank = U[0].shape[1]
        self.lmbda = lmbda
        if not all(np.array([Ui.shape[1] for Ui in U]) == self.rank):
            raise ValueError('Dimension mismatch of factor matrics')
        if lmbda is None:
            self.lmbda = np.ones(self.rank)
    def __eq__(self,other):
        if isinstance(other, ktensor):
            if self.ndim != other.ndim or self.shape != other.shape:
                return False
            return all(
                [(self.U[i] == other.U[i]).all() for i in range(self.ndim)] +
                [(self.lmbda == other.lmbda).all()]
            )
        else:
            # TODO implement __eq__ for tensor_mixins and ndarrays
            raise NotImplementedError()
    def uttkrp(self, U, mode):
        '''
        U:因子矩阵的列表
        按照mode展开张量，并与矩阵U进行Khatri-Rao积
        exampe:
            X:[[1,2],[2,3],[[3,4],[5,6]]]
            X.shape = (1,1,2)
            X.ndim = 3
            X.rank = 2
            
        mode=1:
        '''
        N = self.ndim #二维总个数
        if mode == 1:
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]
        W = np.tile(self.lmbda, 1, R)
        for i in range(mode) + range(mode+1, N):
            W = W*np.dot(self.U[i].T, U[i])
        return dot(self.U[mode], W)
    
    def norm(self):
        '''
        求张量的范数
        '''
        pass
    def innerprod(self, X):
        '''
        求内积
        '''
        pass
    def toarray(self):
        '''
        将张量转换为一个稠密数组
        '''
        pass
    def totensor(self):
        '''
        将张量变成一个稠密张量
        '''
        pass
            