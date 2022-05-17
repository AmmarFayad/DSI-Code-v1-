from sys import ps1
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
import argparse
parser = argparse.ArgumentParser(description = 'ROC EXP')
parser.add_argument('--relation', action='store', type=str, dest='relation', default='linear',
                    help='m')
parser.add_argument('--dim', action='store', type=int, dest='dim', default=10,
                    help='m')

parser.add_argument('--num_points_arr', action='store', dest='num_points_arr', default=[10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000],
                    help='m')
parser.add_argument('--num_slices_arr', action='store', dest='num_slices_arr', default=[10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000],
                    help='m')
args = parser.parse_args()

num_points_arr=args.num_points_arr

num_slices_arr=[]
num_slices=300
dim=args.dim

######## Define X and Y
p=.5
for num_points in num_points_arr: #[num_ponts], [dim], [num_slices]
    X=[]
    Y=[]
    actual_DP=[]
    for i in range(num_points):
        if args.relation=='linear':
            if p>random.random():
                x = np.random.randn(dim, 1)
                z = np.random.randn(dim, 1)
                y=(((np.matmul(np.ones((1,dim)),X)/np.sqrt(dim))*np.ones((dim,1)))+z)/np.sqrt(2)
                dp=1
            else:
                x = np.random.randn(dim, 1)
                y = np.random.randn(dim, 1)

                dp=0
        


            X.append(np.array([x]))
            Y.append(np.array([y]))
            actual_DP.append(dp)

        if args.relation=='common_signal':
            P1=np.random.randn(dim, 2)
            P2=np.random.randn(dim, 2)
            if p>random.random():
                z1 = np.random.randn(dim, 1)
                z2 = np.random.randn(dim, 1)
                v = np.random.randn(2, 1)


                x=np.matmul(P1,v)+z1
                y=np.matmul(P2,v)+z2
                dp=1
            else:
                x = np.random.randn(dim, 1)
                y = np.random.randn(dim, 1)

                dp=0

        if args.relation=='elliptical':
            
            P=np.random.randn(dim, dim)
            if p>random.random():
                x = np.random.rand(dim, 1)
                x/=x.norm(p=2, dim=1, keepdim=True)
                
                y=np.matmul(P,x)
                dp=1
            else:
                x = np.random.rand(dim, 1)
                x/=x.norm(p=2, dim=1, keepdim=True)

                y = np.random.rand(dim, 1)
                y/=y.norm(p=2, dim=1, keepdim=True)

                dp=0

    
     
    
    
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