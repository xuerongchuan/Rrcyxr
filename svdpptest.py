# -*- coding: utf-8 -*-

from configs.config import naisConfig
from readers.naisdataloader import Dataloader, getBatchData
from models.timeSVDplus import timeSVD

config = naisConfig('month')
config.learning_rate = 0.01
config.regU1 = 0.001
config.regU2 = 0.001
config.regU3 = 0.001
config.regI = 0.001
#config.numrows = 50
config.factors = 8
dl = Dataloader(config)
gd = getBatchData(config, dl)
svd = timeSVD(config, dl,gd)
svd.build_graph()

svd.train_and_evaluate()
