# -*- coding: utf-8 -*-

import numpy as np
import time
from  .tensorbase import ktensor
from scipy.linalg import pinv

def als(X, rank, **kwargs):
    '''
    X: 要分解的张量
    rank:因子矩阵的列数及R
    '''
    ainit = 'random'
    dtype = np.float
    N = X.ndim
    maxiter = 500
    #normX = norm()
    
    U = _init(ainit, X, N, rank, dtype)
    fit = 0
    exectimes  = []
    for itr in range(maxiter):
        tic = time.clock()
        fitold = fit
        for n in range(N):
            Unew = X.uttkrp(U,n)
            Y= np.ones((rank, rank), dtype = dtype)
            for i in (list(range(n)) + list(range(n+1, N))):
                Y = Y*np.dot(U[i].T, U[i])
            Unew = Unew.dot(pinv(Y)) #求Y的伪逆
            # 正则化
            if itr == 0:
                lmbda = np.sqrt((Unew**2).sum(axis=0))
            else:
                lmbda = Unew.max(axis=0)
                lmbda[lmbda<1]=1
            U[n] = Unew/lmbda
        P = ktensor(U, lmbda)
    return P

def _init(init, X, N, rank, dtype):
    '''
    初始化因子矩阵
    '''
    Uinit = [None for _ in range(N)]
    if isinstance(init, list):
        Uinit = init
    elif init == 'random':
        for n in range(1,N):
            Uinit[n] = np.array(np.random.rand(X.shape[n], rank), dtype=dtype)
    elif init == 'nvecs':
        for n in range(1, N):
            Uinit[n] = np.array(nvecs(X, n ,rank), dtype = dtype)
    else:
        raise 'Unknown option (init=%s)' % str(init)
    return Uinit