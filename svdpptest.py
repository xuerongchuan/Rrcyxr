# -*- coding: utf-8 -*-

from configs.config import Config
from readers.naisdataloader import Dataloader, getBatchData
from models.timeSVDplus import timeSVD

config = Config()
config.learning_rate = 0.01
#config.numrows = 50
config.factors = 8
dl = Dataloader(config)
gd = getBatchData(config, dl)
svd = timeSVD(config, dl,gd)
svd.build_graph()

svd.train_and_evaluate()
