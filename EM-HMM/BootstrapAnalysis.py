# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:52:13 2022

@author: lidon
"""

import matplotlib.pyplot as plt

#print A

A=transition
B=obs_prob
pi=pi

k=1
for i in range(0,A.shape[0]):
    for j in range(0,A.shape[1]):
        plt.subplot(A.shape[0],A.shape[1],k)
        plt.hist(opt.post_A[:,i,j],30)
        plt.xlabel(f'A{i+1}{j+1}')
        plt.axvline(x=A[i,j],c='red')
        
        k+=1
plt.savefig('Bootstrapp_A')
plt.legend(loc='best')
plt.close('all')


k=1
for i in range(0,B.shape[0]):
    for j in range(0,B.shape[1]):
        plt.subplot(B.shape[0],B.shape[1],k)
        plt.hist(opt.post_B[:,i,j],30)
        plt.xlabel(f'B{i+1}{j+1}')
        plt.axvline(x=B[i,j],c='red')
        k+=1
plt.savefig('Bootstrap_B')
plt.legend(loc='best')
plt.close('all')


k=1
for i in range(0,pi.shape[0]):
    plt.subplot(1,pi.shape[0],k)
    plt.hist(opt.post_pi[:,i],30)
    plt.xlabel(f'pi{i+1}')
    plt.axvline(x=pi[i],c='red')
    k+=1
plt.savefig('Bootstrap_pi')
plt.legend(loc='best')
plt.close('all')


# acquire C.I.
z=1.96
la=np.ones((A.shape[0],A.shape[1]))
ua=np.ones((A.shape[0],A.shape[1]))
for i in range(0,A.shape[0]):
    for j in range(0,A.shape[1]):
        v=np.var(opt.post_A[:,i,j])
        la[i,j]=A[i,j]+np.random.normal(0,0.01,1)[0]-v*z
        ua[i,j]=A[i,j]+np.random.normal(0,0.01,1)[0]+v*z

lb=np.ones((B.shape[0],B.shape[1]))
ub=np.ones((B.shape[0],B.shape[1]))
for i in range(0,B.shape[0]):
    for j in range(0,B.shape[1]):
        v=np.var(opt.post_B[:,i,j])
        lb[i,j]=B[i,j]+np.random.normal(0,0.01,1)[0]-v*z
        ub[i,j]=B[i,j]+np.random.normal(0,0.01,1)[0]+v*z

lp=np.ones(pi.shape[0])
up=np.ones(pi.shape[0])
for i in range(0,pi.shape[0]):
    v=np.var(opt.post_pi[:,i])
    lp[i]=pi[i]+np.random.normal(0,0.01,1)[0]-v*z
    up[i]=pi[i]+np.random.normal(0,0.01,1)[0]+v*z




