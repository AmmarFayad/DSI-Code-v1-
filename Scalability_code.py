from pandas import *
from utils import DSI,calc_TP_FP_rate
import re
import random
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


num_points_arr=[]

num_slices_arr=[]
num_slices=300

######## Define X and Y
for num_points in num_points_arr: #[num_ponts], [dim], [num_slices]
    X=[]
    Y=[]
    actual_DP=[]
    for i in range(num_points):
        
        x= random.uniform(0,1)

        y=x
        dp=1
    


        X.append(np.array([x]))
        Y.append(np.array([y]))
        actual_DP.append(dp)
    X=np.array(X)
    Y=np.array(Y)







    predictions=np.array([])
    

    
    DSI_val= DSI(X,Y, num_slices)
    predictions=np.append(predictions,[DSI_val.detach().numpy()])

    # predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))


    tp_rates = []
    fp_rates = []

    # Define probability thresholds to use, between 0 and 1
    probability_thresholds = np.linspace(0,1,num=100)

    # Find true positive / false positive rate for each threshold
    for p in probability_thresholds:
        
        y_test_preds = []
        
        for prob in predictions:
            if prob > p:
                y_test_preds.append(1)
            else:
                y_test_preds.append(0)
                
        tp_rate, fp_rate = calc_TP_FP_rate(actual_DP, y_test_preds)
            
        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)

    # actual=np.ones(1)
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # print(roc_auc)

    AUC=auc(fp_rates, tp_rates)