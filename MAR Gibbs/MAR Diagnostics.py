# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:19:29 2021

@author: s1155151972
"""


import matplotlib.pyplot as plt
import numpy


# import Sampling of A (transition matrix)
post_A1=np.load('Experiment1/Post_A.npy')
post_A2=np.load('Experiment2/Post_A.npy')
post_A3=np.load('Experiment3/Post_A.npy')
post_A4=np.load('Experiment4/Post_A.npy')

# import Sampling of B (observation matrix)
post_B1=np.load('Experiment1/Post_B.npy')
post_B2=np.load('Experiment2/Post_B.npy')
post_B3=np.load('Experiment3/Post_B.npy')
post_B4=np.load('Experiment4/Post_B.npy')

# import estimated latent sequence
latent_seq1=np.load('Experiment1/latent_seq.npy')
latent_seq2=np.load('Experiment2/latent_seq.npy')
latent_seq3=np.load('Experiment3/latent_seq.npy')
latent_seq4=np.load('Experiment4/latent_seq.npy')

# import log probability of the joint distribution
log_prob1=np.loadtxt('Experiment1/log_prob.txt')
log_prob2=np.loadtxt('Experiment2/log_prob.txt')
log_prob3=np.loadtxt('Experiment3/log_prob.txt')
log_prob4=np.loadtxt('Experiment4/log_prob.txt')

# import data and hidden sequqnce
data=np.load('Experiment1/data.npy')
hidden_seq=np.load('Experiment1/TrueHidden.npy')

# Trace plot analysis of A11
plt.plot(np.arange(0,len(post_A1)),post_A1[:,0,0],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_A2)),post_A2[:,0,0],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_A3)),post_A3[:,0,0],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_A4)),post_A4[:,0,0],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_A1)),np.repeat(transition[0,0],len(post_A1)),'black',label='True Value')
plt.title('Sampling result of A11')
plt.legend(loc='best')
plt.xlabel('iteration')


# Trace plot analysis of A22
plt.plot(np.arange(0,len(post_A1)),post_A1[:,1,1],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_A2)),post_A2[:,1,1],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_A3)),post_A3[:,1,1],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_A4)),post_A4[:,1,1],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_A1)),np.repeat(transition[1,1],len(post_A1)),'black',label='True Value')
plt.title('Sampling result of A22')
plt.legend(loc='best')
plt.xlabel('iteration')


# Trace plot analysis of A33
plt.plot(np.arange(0,len(post_A1)),post_A1[:,2,2],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_A2)),post_A2[:,2,2],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_A3)),post_A3[:,2,2],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_A4)),post_A4[:,2,2],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_A1)),np.repeat(transition[2,2],len(post_A1)),'black',label='True Value')
plt.title('Sampling result of A33')
plt.legend(loc='best')
plt.xlabel('iteration')


# Trace plot analysis of A44
plt.plot(np.arange(0,len(post_A1)),post_A1[:,3,3],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_A2)),post_A2[:,3,3],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_A3)),post_A3[:,3,3],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_A4)),post_A4[:,3,3],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_A1)),np.repeat(transition[3,3],len(post_A1)),'black',label='True Value')
plt.title('Sampling result of A44')
plt.legend(loc='best')
plt.xlabel('iteration')



# Trace plot analysis of B11
plt.subplot(2,2,1)
plt.plot(np.arange(0,len(post_B1)),post_B1[:,0,0],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[0,0],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(np.arange(0,len(post_B2)),post_B2[:,0,0],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[0,0],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(np.arange(0,len(post_B3)),post_B3[:,0,0],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[0,0],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(np.arange(0,len(post_B4)),post_B4[:,0,0],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[0,0],len(post_B1)),'black',label='True Value')

plt.xlabel('iteration')
plt.legend(loc='best')



# Trace plot analysis of B23
plt.subplot(2,2,1)
plt.plot(np.arange(0,len(post_B1)),post_B1[:,1,2],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[1,2],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(np.arange(0,len(post_B2)),post_B2[:,1,2],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[1,2],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(np.arange(0,len(post_B3)),post_B3[:,1,2],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[1,2],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(np.arange(0,len(post_B4)),post_B4[:,1,2],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[1,2],len(post_B1)),'black',label='True Value')

plt.xlabel('iteration')
plt.legend(loc='best')

# Trace plot analysis of B35
plt.subplot(2,2,1)
plt.plot(np.arange(0,len(post_B1)),post_B1[:,2,4],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[2,4],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(np.arange(0,len(post_B2)),post_B2[:,2,4],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[2,4],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(np.arange(0,len(post_B3)),post_B3[:,2,4],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[2,4],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(np.arange(0,len(post_B4)),post_B4[:,2,4],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[2,4],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

# Trace plot analysis of B43
plt.subplot(2,2,1)
plt.plot(np.arange(0,len(post_B1)),post_B1[:,3,2],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[3,2],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(np.arange(0,len(post_B2)),post_B2[:,3,2],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[3,2],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(np.arange(0,len(post_B3)),post_B3[:,3,2],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[3,2],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(np.arange(0,len(post_B4)),post_B4[:,3,2],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[3,2],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

# Trace plot analysis of B55
plt.subplot(2,2,1)
plt.plot(np.arange(0,len(post_B1)),post_B1[:,4,4],'r',label='Experiment1')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[4,4],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(np.arange(0,len(post_B2)),post_B2[:,4,4],'g',label='Experiment2')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[4,4],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(np.arange(0,len(post_B3)),post_B3[:,4,4],'b',label='Experiment3')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[4,4],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(np.arange(0,len(post_B4)),post_B4[:,4,4],'purple',label='Experiment4')
plt.plot(np.arange(0,len(post_B1)),np.repeat(obs_prob[4,4],len(post_B1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')



#Trace plot analysis of the joint log pdf
h=latent_seq1.copy()
for i in range(0,h.shape[0]):
    for j in range(0,h.shape[1]):
        if h[i,j]!='None':
            h[i,j]=hidden_seq[i,j]

true_log_p=p_evaluator(transition,obs_prob,h,data)


plt.subplot(2,2,1)
plt.plot(np.arange(0,len(log_prob1)),log_prob1,'r',label='Experiment1')
plt.plot(np.arange(0,len(log_prob1)),np.repeat(true_log_p,len(log_prob1)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(np.arange(0,len(log_prob2)),log_prob2,'g',label='Experiment2')
plt.plot(np.arange(0,len(log_prob1)),np.repeat(true_log_p,len(log_prob2)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(np.arange(0,len(log_prob3)),log_prob3,'b',label='Experiment3')
plt.plot(np.arange(0,len(log_prob1)),np.repeat(true_log_p,len(log_prob3)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(np.arange(0,len(log_prob4)),log_prob4,'purple',label='Experiment4')
plt.plot(np.arange(0,len(log_prob1)),np.repeat(true_log_p,len(log_prob4)),'black',label='True Value')
plt.xlabel('iteration')
plt.legend(loc='best')

# Display accuracy



data1=np.load('Experiment1/data.npy')
hidden_seq1=np.load('Experiment1/TrueHidden.npy')
data2=np.load('Experiment2/data.npy')
hidden_seq2=np.load('Experiment2/TrueHidden.npy')
data3=np.load('Experiment3/data.npy')
hidden_seq3=np.load('Experiment3/TrueHidden.npy')
data4=np.load('Experiment4/data.npy')
hidden_seq4=np.load('Experiment4/TrueHidden.npy')

acc1=np.sum(latent_seq1==hidden_seq1)/np.sum(data1!='None')
acc2=np.sum(latent_seq2==hidden_seq2)/np.sum(data2!='None')
acc3=np.sum(latent_seq3==hidden_seq3)/np.sum(data3!='None')
acc4=np.sum(latent_seq4==hidden_seq4)/np.sum(data4!='None')

# Compute the accuracy according to subtype

# Accuracy of each Subtype:

acc1=[]
for k in hidden_state:
    acc1.append(np.sum((latent_seq1==hidden_seq1)*(latent_seq1==k))/np.sum(h==k))
    
acc2=[]
for k in hidden_state:
    acc2.append(np.sum((latent_seq2==hidden_seq2)*(latent_seq2==k))/np.sum(h==k))

acc3=[]
for k in hidden_state:
    acc3.append(np.sum((latent_seq3==hidden_seq2)*(latent_seq3==k))/np.sum(h==k))

acc4=[]
for k in hidden_state:
    acc4.append(np.sum((latent_seq4==hidden_seq2)*(latent_seq4==k))/np.sum(h==k))















