# -*- coding: utf-8 -*-

from configs.config import naisConfig
from readers.naisdataloader import Dataloader, getBatchData
from models.nais import NAIS

config = naisConfig()
dl = Dataloader(config)
dl.init_data()
gd = getBatchData(config, dl)
nais = NAIS(config, gd)
nais.build_graph()

nais.train_and_evaluate()
