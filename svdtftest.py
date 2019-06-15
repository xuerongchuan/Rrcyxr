# -*- coding: utf-8 -*-

from configs.config import svdConfig
from readers.implicitReader import Dataloader
from models.SVDtf import SVD

config = svdConfig()
config.learning_rate = 0.01
#config.numrows = 50
config.factors = 16
dl = Dataloader(config)
svd = SVD(config, dl)
svd.build_graph()

svd.train_and_evaluate()
