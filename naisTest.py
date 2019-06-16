# -*- coding: utf-8 -*-

from configs.config import naisConfig
from readers.naisdataloader import Dataloader, getBatchData
from models.nais import NAIS

config = naisConfig('month')
dl = Dataloader(config)
gd = getBatchData(config, dl)
nais = NAIS(config, gd)
nais.build_graph()
try:
    nais.train_and_evaluate()
except Exception as e:
    print(e)