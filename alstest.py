# -*- coding: utf-8 -*-

from models.MFals import ALS
from configs.config import Config
from readers.naisdataloader import Dataloader

config = Config()
dl = Dataloader(config)
als = ALS(config, dl)
als.train_and_evaluate()
