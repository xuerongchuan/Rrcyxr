from configs.config import naisConfig
from readers.naisdataloader import Dataloader, getBatchData

config = naisConfig()
dl = Dataloader(config)
dl.init_data()
gd = getBatchData(config, dl)

print(next(gd.getTrainBatches())[0])


