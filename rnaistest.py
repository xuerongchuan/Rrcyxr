# -*- coding: utf-8 -*-
from configs.config import naisConfig
from readers.reader import Dataloader, getBatchData
from models.rnais import RNAIS

config = naisConfig('rating')
dl = Dataloader(config)
gd = getBatchData(config, dl)
tais = RNAIS(config, gd)
tais.build_graph()
tais.train_and_evaluate()