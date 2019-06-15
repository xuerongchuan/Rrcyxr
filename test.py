# -*- coding: utf-8 -*-
import sys
from models.userCF import UserCF
from models.itemCF import ItemCF
from configs.config import Config
import numpy as np
import time

if __name__ == '__main__':
    config = Config()
    config.rating_cv_path = sys.argv[1]
    config.dataset_name = sys.argv[2]
    config.k_fold_num = int(sys.argv[3])
    model = ItemCF(config)
    rmse_list, mae_list = [], []
    for tmp in range(config.k_fold_num):
        model.init_model(tmp)
        ti = time.time()
        rmse, mae = model.predict_model()
        to = time.time()
        print('消耗时间:',(to-ti))
        rmse_list.append(rmse)
        mae_list.append(mae)
    print('rmse:', np.mean(rmse_list),'\n', 'mae:', \
          np.mean(mae_list))
    
    