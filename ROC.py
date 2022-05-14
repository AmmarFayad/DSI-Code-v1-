from turtle import onclick, shape
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from pandas import *
from utils import DSI
import re
import random
import math
args=[]
################### data sets generation
X=[]
Y=[]
p=args.noise_ratio
for i in args.num_points:
    if args.relation=='linear':
        if p>random.random():

            x= random.uniform(0,1)
            y= random.uniform(0,1)
            
        else:
            x= random.uniform(0,1)

            y=x
    

    if args.relation=='Parabolic':
        if p>random.random():

            x= random.uniform(0,1)
            y= random.uniform(0,1)
            
        else:
            x= random.uniform(0,1)

            y=x**2
    
    if args.relation=='Ellipse':
        theta = np.arange(0, 360, .5)
        if p>random.random():

            x= random.uniform(0,1)
            y= random.uniform(0,1)
            
        else:
            x= .9*np.cos(theta[i])

            y=.9*np.cos(theta[i])
        
    if args.relation=='Sinusoidal':
        if p>random.random():

            x= random.uniform(0,1)
            y= random.uniform(0,1)
            
        else:
            x= random.uniform(0,1)

            y=np.cos(x)
    
    if args.relation=='Two Sinusoidals':
        if p>random.random():

            x= random.uniform(0,1)
            y= random.uniform(0,1)
            
        else:
            if random.random()>.5:
                x= random.uniform(0,1)

                y=np.cos(x)
            else:
                x= random.uniform(0,1)

                y=np.cos(x+.5)

    if args.relation=='ZigZag':
        k=args.k
        if p>random.random():

            x= random.uniform(0,1)
            y= random.uniform(0,1)
            
        else:
            x= random.uniform(0,1)

            y=(k*x-math.floor(k*x))*(-1+2*(math.floor(k*x)%2))-(math.floor(k*x)%2)
        
    if args.relation=="Epicycloid":
        theta = np.arange(0, 360, .5)
        k=args.k
        if p>random.random():
            x= random.uniform(-k-2,k+1)
            y= random.uniform(-k-2,k+1)
        else:
            x=(k+1) *np.cos(theta[i]) - np.cos((k+1) *theta[i])

            y=(k+1) *np.sin(theta[i]) - np.sin((k+1) *theta[i])

    if args.relation=="Hypocycloid":
        theta = np.arange(0, 360, .5)
        k=args.k
        if p>random.random():
            x= random.uniform(-k-2,k+1)
            y= random.uniform(-k-2,k+1)
        else:
            x=(k-1) *np.cos(theta[i]) - np.cos((k-1) *theta[i])

            y=(k-1) *np.sin(theta[i]) - np.sin((k-1) *theta[i])



    X.append(x)
    Y.append(y)

###############################################################

d=2
n=100 #1000 to produce acurate joint distributions
ns=d
nd=50 #50
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

for _ in range(nd):
    X = np.random.randn(d, n)
    Y = np.random.randn(d, n)
    DSI_val= DSI(X,Y, args.num_slices)
    predictions=np.append(predictions,[DSI_val])

# predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))

actual=np.ones(nd)
actual=np.append(actual,np.zeros(nd))
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

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