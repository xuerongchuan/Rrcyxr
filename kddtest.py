from configs.svdConfig import Config
from readers.mlReader import Dataloader
from models.kdd import KDD

config = Config()
dl =  Dataloader(config)
model = KDD(config, dl)
model.evaluate()