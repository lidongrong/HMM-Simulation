# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:54:09 2021

@author: s1155151972
"""


import matplotlib.pyplot as plt

# plot the trace plot of the posterior
# omit the initial guess to ensure the scale
plt.plot(np.arange(1,len(log_prob)),log_prob[1:])

# Trace plot of part of the elements in A
a22=post_A[:,1,1]
plt.plot(np.arange(0,len(a22)),a22)

a33=post_A[:,2,2]
plt.plot(np.arange(0,len(a33)),a33)

a44=post_A[:,3,3]
plt.plot(np.arange(0,len(a44)),a44)


# Trace plot of selected elements in B (B[0,0] -> B[3,3]->B[5,4])
b11=post_B[:,0,0]
plt.plot(np.arange(0,len(b11)),b11)

b22=post_B[:,1,1]
plt.plot(np.arange(0,len(b22)),b22)

b33=post_B[:,2,2]
plt.plot(np.arange(0,len(b33)),b33)

b44=post_B[:,3,3]
plt.plot(np.arange(0,len(b44)),b44)

b54=post_B[:,4,3]
plt.plot(np.arange(0,len(b54)),b54)

#Acquire the likelihood of the model with true parameters & latent sequence
for i in range(0,I.shape[0]):
    for j in range(0,I.shape[1]):
        if I[i,j]!='None':
            I[i,j]=Sampling.hidden_data[i,j]
