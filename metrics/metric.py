import math

class Metric(object):
    
    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2]- entry[3])
            count += 1
        if count == 0:
            return error
        return float(error) / count
    
    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2]- entry[3])**2
            count += 1
        if count == 0:
            return error
        return math.sqrt(float(error)) / count
    @staticmethod
    def precision_and_recall(res):
        hit = 0
        count = 0
        items = res[1]
        for entry in res[0]:
            if entry in items:
                hit+=1
            count += 1
        if count == 0:
            return error
        return hit/len(res[0]), hit/len(res[1])