# -*- coding: utf-8 -*-
import math 
def common(x1, x2):
    common = (x1 != 0) &(x2 != 0)
    new_x1 = x1[common]
    new_x2 = x2[common]
    return new_x1, new_x2

def pearson(x1, x2):
    new_x1, new_x2 = common(x1, x2)
    
    ind1 = new_x1 >0
    ind2 = new_x2 >0
    try:
        mean_x1 = float(new_x1.sum()) / ind1.sum()
        mean_x2 = float(new_x2.sum()) / ind2.sum()
        new_x1 = new_x1 - mean_x1
        new_x2 = new_x2 - mean_x2
        sum = new_x1.dot(new_x2)
        denom = math.sqrt((new_x1.dot(new_x1)) * (new_x2.dot(new_x2)))
        return float(sum) / denom
    except ZeroDivisionError:
        return 0

def pearson_sp(x1, x2):
    '''针对稀疏的字典类型的x1和x2'''
    common = set(x1.keys()) & set(x2.keys())
    if len(common) == 0:
        return 0
    ratingList1, ratingList2 = [], []
    for i in common:
        ratingList1.append(x1[i])
        ratingList2.append(x2[i])
    if len(ratingList1) == 0 or len(ratingList2) == 0:
        return 0
    avg1 = sum(ratingList1)/len(ratingList1)
    avg2 = sum(ratingList2)/len(ratingList2)
    mult = 0.0
    sum1 = 0.0
    sum2 = 0.0
    for i in range(len(ratingList1)):
        mult += (ratingList1[i]-avg1)* (ratingList2[i] - avg2)
        sum1 += pow(ratingList1[i]-avg1, 2)
        sum2 += pow(ratingList2[i]-avg2, 2)
    if sum1 ==0 or sum2 ==0:
        return 0
    return mult /(math.sqrt(sum1)*math.sqrt(sum2))