# -*- coding: utf-8 -*-
from configs.config import naisConfig
from readers.naisdataloader import Dataloader, getBatchData
from models.tais import TAIS

config = naisConfig()
config.mode ='day'
dl = Dataloader(config)
gd = getBatchData(config, dl)
tais = TAIS(config, gd)
tais.build_graph()
tais.train_and_evaluate()

