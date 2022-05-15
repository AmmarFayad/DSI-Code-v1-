from turtle import onclick, shape
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from pandas import *
from utils import DSI,calc_TP_FP_rate
import re
import random
import math
import argparse
parser = argparse.ArgumentParser(description = 'ROC EXP')

parser.add_argument('--noise_ratios', action='store', dest='noise_ratios', default=[.5,.6,.7,.8,.9,1],
                    help='m')

parser.add_argument('--relation', action='store', type=str, dest='relation', default='linear',
                    help='m')

parser.add_argument('--num_slices', action='store', dest='num_slices', default=300,
                    help='m')

parser.add_argument('--num_points', action='store', type=int, dest='num_points', default=700,
                    help='m')

args = parser.parse_args()
################### data sets generation

for p in args.noise_ratios:
    
    X=[]
    Y=[]
    actual_DP=[]
    for i in range(args.num_points):
        if args.relation=='linear':
            if p<random.random():

                x= random.uniform(0,1)
                y= random.uniform(0,1)
                dp=0
                
            else:
                x= random.uniform(0,1)

                y=x
                dp=1
        

        if args.relation=='Parabolic':
            if p<random.random():

                x= random.uniform(0,1)
                y= random.uniform(0,1)
                dp=0
                
            else:
                x= random.uniform(0,1)

                y=x**2
                dp=1
        
        if args.relation=='Ellipse':
            theta = np.arange(0, 360, .5)
            if p<random.random():

                x= random.uniform(0,1)
                y= random.uniform(0,1)
                dp=0
                
            else:
                x= .9*np.cos(theta[i])

                y=.9*np.cos(theta[i])
                dp=1
            
        if args.relation=='Sinusoidal':
            if p<random.random():

                x= random.uniform(0,1)
                y= random.uniform(0,1)
                dp=0
                
            else:
                x= random.uniform(0,1)

                y=np.cos(x)
                dp=1
        
        if args.relation=='Two Sinusoidals':
            if p<random.random():

                x= random.uniform(0,1)
                y= random.uniform(0,1)
                dp=0
                
            else:
                if random.random()>.5:
                    x= random.uniform(0,1)

                    y=np.cos(x)
                else:
                    x= random.uniform(0,1)

                    y=np.cos(x+.5)
                    dp=1

        if args.relation=='ZigZag':
            k=args.k
            if p<random.random():

                x= random.uniform(0,1)
                y= random.uniform(0,1)
                dp=0
                
            else:
                x= random.uniform(0,1)

                y=(k*x-math.floor(k*x))*(-1+2*(math.floor(k*x)%2))-(math.floor(k*x)%2)
                dp=1
            
        if args.relation=="Epicycloid":
            theta = np.arange(0, 360, .5)
            k=args.k
            if p<random.random():
                x= random.uniform(-k-2,k+1)
                y= random.uniform(-k-2,k+1)
                dp=0
            else:
                x=(k+1) *np.cos(theta[i]) - np.cos((k+1) *theta[i])

                y=(k+1) *np.sin(theta[i]) - np.sin((k+1) *theta[i])
                dp=1

        if args.relation=="Hypocycloid":
            theta = np.arange(0, 360, .5)
            k=args.k
            if p<random.random():
                x= random.uniform(-k-2,k)
                y= random.uniform(-k-2,k)
                dp=0
            else:
                x=(k-1) *np.cos(theta[i]) - np.cos((k-1) *theta[i])

                y=(k-1) *np.sin(theta[i]) - np.sin((k-1) *theta[i])
                dp=1
        
        if args.relation=="Rose_Curve":
            theta = np.arange(0, 360, .5)
            k=args.k
            if p<random.random():
                x= random.uniform(-1,+1)
                y= random.uniform(-1,+1)
                dp=0
            else:
                x=np.cos(theta[i])* np.cos((k) *theta[i])

                y=np.sin(theta[i]) * np.sin((k) *theta[i])
                dp=1



        X.append(np.array([x]))
        Y.append(np.array([y]))
        actual_DP.append(dp)
    X=np.array(X)
    Y=np.array(Y)
    # actual_DP=np.array(actual_DP)

    ###############################################################

    
    predictions=np.array([])
    # for _ in range(nd):
    #     X = np.random.randn(d, n)
    #     Z = np.random.randn(d, n)
    #     Y1=(((np.matmul(np.ones((1,d)),X)/np.sqrt(d))*np.ones((d,1)))+Z)/np.sqrt(2)
    #     Y=(X+Z)/np.sqrt(2)
    #     dist = Table().domain(Y1,X)
    #     ax=ay=np.empty(shape=[0, n])
    #     for i in range(ns):
    #         r=np.random.randint(d)
    #         a=dist.row(r)
    #         ax=np.vstack([ax,a.X])
    #         ay=np.vstack([ay,a.Y])
    #     DSI_val = sliced_MI(ax,ay, 100)
    #     predictions=np.append(predictions,[DSI_val])

    
    DSI_val= DSI(X,Y, args.num_slices)
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

#dist = dist.probabilities(np.ones(d*d)/(d*d))
#a=dist.sample()
#y=a.iloc[0].name
#x = list(a)[np.random.randint(d)]
#x=re.findall('\d*\.?\d+',x)
#y=re.findall('\d*\.?\d+',y)
#print(type(y[0]))
#print(dist.iloc[1].name)

'''plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'blue', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'m--')
plt.xlim([0,1])
plt.ylim([0,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''