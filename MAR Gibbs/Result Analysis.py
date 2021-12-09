# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:55:01 2021

@author: s1155151972
"""


# The result analysis script is desinged to analyze all results in one-shot



import os
import matplotlib.pyplot as plt
import math

# num specifies the total experiments that has been run
num=8

# Save the data
for i in range(0,num):
    os.mkdir(f'Experiment{i}')
    # Save the results
    np.save(f'Experiment{i}/Post_A.npy',out_obj[i].post_A)
    np.save(f'Experiment{i}/Post_B.npy',out_obj[i].post_B)
    np.save(f'Experiment{i}/latent_seq.npy',out_obj[i].latent_seq)
    np.savetxt(f'Experiment{i}/log_prob.txt',out_obj[i].log_prob)
    np.save(f'Experiment{i}/data.npy',out_obj[i].data)
    np.save(f'Experiment{i}/TrueHidden.npy',out_obj[i].true_hidden)


#Alternative choice of transition matrix
transition=np.array([[0.4,0.6,0,0,0],[0,0.3,0.7,0,0],[0,0,0.9,0.1,0],[0,0,0,0.8,0.2],[0,0,0,0,1]]
    )

                    
# Read the data
output=[]

for i in range(0,num):
    post_A=np.load(f'Experiment{i}/Post_A.npy')
    post_B=np.load(f'Experiment{i}/Post_B.npy')
    latent_seq=np.load(f'Experiment{i}/latent_seq.npy')
    log_prob=np.loadtxt(f'Experiment{i}/log_prob.txt')
    data=np.load(f'Experiment{i}/data.npy')
    hidden_seq=np.load(f'Experiment{i}/TrueHidden.npy')
    
    output.append(Out(data,post_A,post_B,latent_seq,log_prob,hidden_seq))


output=output[0:6]
num=6


color_bar=['red','blue','green','pink','k','violet','gold','brown','c','m']


# Paint the Trace Plot of a11
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,0,0],color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[0,0],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')

# Paint the Trace Plot of a22
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,1,1],color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[1,1],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')

# Paint the Trace Plot of a33
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,2,2],color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[2,2],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')

# Paint the Trace Plot of a44
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,3,3],color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[3,3],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')



# Paint the Trace Plot of b11
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_B[:,1,1],color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_B)),np.repeat(obs_prob[1,1],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')

# Paint the Trace Plot of b33
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_B[:,2,2],color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_B)),np.repeat(obs_prob[2,2],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')

# Paint the Trace Plot of b54
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_B[:,4,3],color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_B)),np.repeat(obs_prob[4,3],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')


# Paint the Trace Plot log-prob

# Acquire true hidden seq
h=output[1].latent_seq.copy()
for i in range(0,h.shape[0]):
    for j in range(0,h.shape[1]):
        if h[i,j]!='None':
            h[i,j]=output[1].true_hidden[i,j]

# compute true log prob
true_log_p=p_evaluator(transition,obs_prob,h,output[0].data)

for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].log_prob)),output[i].log_prob,color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].log_prob)),np.repeat(true_log_p,len(output[i].log_prob)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')


# Accuracy
for i in range(0,len(output)):
    acc=np.sum(output[i].latent_seq==output[i].true_hidden)/np.sum(output[i].data!='None')
    print(acc)





