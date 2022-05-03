# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:04:46 2022

@author: lidon
"""

import numpy as np
from ZMARGibbs import*
from HMM import*
from SeqSampling import*
from EMHMM import*
import math
from PMAP import*
import matplotlib.pyplot as plt


transition=np.array([[0.6,0.2,0.1,0.05,0.05],[0.05,0.6,0.2,0.1,0.05],[0.05,0.05,0.6,0.2,0.1],
                      [0.1,0.05,0.05,0.6,0.2],[0.2,0.1,0.05,0.05,0.6]])
'''
transition=np.array([[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],
                     [0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2]])
'''
state=np.array(['0','1','2','3','4'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow','red','orange'])
obs_prob=np.array([[0.9,0.07,0.01,0.01,0.01],[0.01,0.9,0.07,0.01,0.01],[0.01,0.01,0.9,0.07,0.01],
                     [0.01,0.01,0.01,0.9,0.07],[0.07,0.01,0.01,0.01,0.9]
     ])
#obs_prob=np.eye(5)
# obs_prob=np.array([[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],
#                       [0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2]])

pi=np.linalg.matrix_power(transition,100)[0]



# calculate the entropy
def entropy(x):
    return -np.dot(x[x>0],np.log2(x[x>0]))

# calculate hamming epsilon ball
# T: length
# eps: epsilon
# z: alphabet size
def eps_ball(T,eps,z):
    maximum=np.floor(T*eps)
    maximum=int(maximum)
    s=0
    for k in range(0,maximum):
        s=s+(math.factorial(T)/(math.factorial(k)*math.factorial(T-k)))*((z-1)**k)
    return s

#entropy of a markov chain
# T: length
def MC_entropy1(T,A,B,pi):
    s=entropy(pi)
    for k in range(1,T):
        prob=np.dot(pi,np.linalg.matrix_power(A,k))
        ent=np.array([entropy(A[i]) for i in range(0,A.shape[0])])
        s=s+np.dot(prob,ent)
    return s


#entropy of a markov chain
# T: length
def MC_entropy(T,A,B,pi):
    s=entropy(pi)
    for k in range(1,T):
        prob=pi
        ent=np.array([entropy(A[i]) for i in range(0,A.shape[0])])
        s=s+np.dot(prob,ent)
    return s


eps=0.15
T=10
z=transition.shape[0]
# obs rate
#q=0.9
# p=(MC_entropy(T,transition,obs_prob,pi)-1-T*q*(np.log2(3)-entropy(obs_prob[0]))-np.log2(eps_ball(T,eps,z)))
# p=p/np.log2((3**T)/eps_ball(T,eps,z)-1)


prec=[]

# lower bound
# q: observation proportion
for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    p=(MC_entropy1(T,transition,obs_prob,pi)-1-T*q*(entropy(np.dot(pi,obs_prob))-entropy(obs_prob[0]))-np.log2(eps_ball(T,eps,z)))
    p=p/np.log2((transition.shape[0]**T)/eps_ball(T,eps,z)-1)
    prec.append(p)

# upper bound
ubound=[]
for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    p=1/T*(T-T*q*(1/2**(entropy(pi)+entropy(obs_prob[0])-entropy(np.dot(pi,obs_prob))))-T*(1-q)*(1/2**(entropy(np.dot(pi,obs_prob)))))
    ubound.append(p)


prec.reverse()
ubound.reverse()
pred_prec=[]
pred_acc=[]
for rate in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    print('rate:',rate)
    A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
    data=data[:,0:T]
    hidden_data=hidden_data[:,0:T]
    #data=state_missing(data,hidden_data,hidden_state[0],0.3)
    z1=PMAP_dataset(data,transition,obs_prob,pi,hidden_state,obs_state)
    a=acc(z1,hidden_data)
    pred_prec.append(sum(a<1-eps)/5000)
    pred_acc.append(sum(1-a)/5000)

plt.plot(prec,'*-',label='Theoretical Lower Bound')
plt.plot(pred_prec,'*-',label=f'Empirical probability')
#plt.plot(ubound,'*-',label='Theoretical Upper Bound')
#plt.plot(pred_acc,'*-',label='empirical error')
plt.xlabel('Missing Rate')
plt.legend(loc='best')
