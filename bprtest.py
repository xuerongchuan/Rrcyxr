# -*- coding: utf-8 -*-

from models.BPR import BPR
from readers.naisdataloader import Dataloader
from configs.config import Config

config = Config()
dl = Dataloader(config)
bpr = BPR(config, dl)
bpr.train_and_evaluate()
          
          