from readers import ratingReader
from models.userCF import UserCF
from configs.svdConfig import Config

config = Config()
model = UserCF(config)
model.init_model()
model.evaluate()