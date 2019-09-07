import math

class Metric(object):
    '''
    res :结果列表，形式为
    用户id， 商品id， 真实值，预测值
    '''
    # 评估评分预测准确度，MAE，RMSE
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
    def ratingMeasure(res):
        measure = []
        mae = Metric.MAE(res)
        measure.append('MAE:'+str(mae)+'\n')
        rmse = Metric.RMSE(res)
        measure.append('RMSE' + str(rmse)+'\n')
        return measure

    # evaluate for ranking
    #res: N recommendation for every user
    # origin:  user history interact items with variable length
    @staticmethod
    def hits(origin, res):
        '''保存每个用户的推荐命中个数，
        方便后面的计算'''
        hitCount = {}
        for user in origin:
            #origin keeps the original label
            items = origin[user].keys()
            predicted = [item[0] for item in res[user]]
            hitCount = len(set(items)&set(predicted))
        return hitCount
    @staticmethod
    def presicion(hits, N):
        prec = sum([hits[user] for user in hits])
        return float(prec)/(len(hits)*N)
    @staticmethod
    def recall(hits, origin):
        res = sum(hits[user]/len(origin[user]) for \
            user in hits)
        return float(res)/len(hits)
    @staticmethod
    def F1(prec, recal):
        if (prec+recal):
            return 2*prec*recal/(prec+recal)
        else:
            return 0.0
    @staticmethod
    def MAP(res, origin, N):
        #res has been reverse sorted
        map = 0.0
        for user in res:
            num = 0
            ans = 0
            rec_u = res[user]
            for ind, (item,_) in enumerate(rec_u):
                if item in origin[user]:
                    num += 1
                    ans += (num+1.0)/(ind+1.0)
            map += ans/min(len(origin[user], N))
        return float(ans)/len(res)
    @staticmethod
    def HR(origin, hits):
        numer = 0.0
        denom = 0.0
        for user in hits:
            numer += hits[user]
            denom += len(origin[user])
        return float(numer/denom)
    @staticmethod
    def NDCG(origin, res, N):
        ndcg = 0.0
        for user in res:
            dcg = 0.0
            idcg = 0.0
            #这里相关性只考虑0，1相关，即只有相关和不相关，两种情况
            for idx, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    dcg += 1.0/math.log2(idx+2.0)
            for idx in range(min(N, len(origin[user]))):
                idcg += 1.0/math.log2(idx+2.0)
            ndcg += dcg/idcg
        return ndcg/len(res)

    @staticmethod
    def rankingMeasure(origin, res, N):
        measure = []
        for n in N:
            predicted = {}
            for user in res:
                predicted[user] = res[user][:n]
            indicators = []
            if len(origin) != len(predicted):
                print 'The Lengths of test set and predicted set are not match!'
                exit(-1)
            hits = Measure.hits(origin, predicted)
            prec = Measure.precision(hits, n)
            indicators.append('Precision:' + str(prec) + '\n')
            recall = Measure.recall(hits, origin)
            indicators.append('Recall:' + str(recall) + '\n')
            F1 = Measure.F1(prec, recall)
            indicators.append('F1:' + str(F1) + '\n')
            MAP = Measure.MAP(origin, predicted, n)
            indicators.append('MAP:' + str(MAP) + '\n')
            NDCG = Measure.NDCG(origin, predicted, n)
            indicators.append('NDCG:' + str(NDCG) + '\n')
            # AUC = Measure.AUC(origin,res,rawRes)
            # measure.append('AUC:' + str(AUC) + '\n')
            measure.append('Top ' + str(n) + '\n')
            measure += indicators
        return measure


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