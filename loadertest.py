from configs.config import naisConfig
from readers.naisdataloader import Dataloader, getBatchData

config = naisConfig('month')
dl = Dataloader(config)
gd = getBatchData(config, dl)
i=0
for uData in dl.trainset():
	print(uData)
	if i >=1:
		break
	i+=1

