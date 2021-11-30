# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:56:06 2021

@author: s1155151972
"""


import os

os.mkdir('Experiment1')
os.mkdir('Experiment2')
os.mkdir('Experiment3')
os.mkdir('Experiment4')





# Save the results
np.save('Experiment1/Post_A.npy',post_A)
np.save('Experiment1/Post_B.npy',post_B)
np.save('Experiment1/latent_seq.npy',latent_seq)
np.savetxt('Experiment1/log_prob.txt',log_prob)
np.save('Experiment1/data.npy',data)
np.save('Experiment1/TrueHidden.npy',Sampling.hidden_data)


np.save('Experiment2/Post_A.npy',post_A1)
np.save('Experiment2/Post_B.npy',post_B1)
np.save('Experiment2/latent_seq.npy',latent_seq1)
np.savetxt('Experiment2/log_prob.txt',log_prob1)
np.save('Experiment2/data.npy',data)
np.save('Experiment2/TrueHidden.npy',Sampling.hidden_data)

np.save('Experiment3/Post_A.npy',post_A2)
np.save('Experiment3/Post_B.npy',post_B2)
np.save('Experiment3/latent_seq.npy',latent_seq2)
np.savetxt('Experiment3/log_prob.txt',log_prob2)
np.save('Experiment3/data.npy',data)
np.save('Experiment3/TrueHidden.npy',Sampling.hidden_data)


np.save('Experiment4/Post_A.npy',post_A4)
np.save('Experiment4/Post_B.npy',post_B4)
np.save('Experiment4/latent_seq.npy',latent_seq4)
np.savetxt('Experiment4/log_prob.txt',log_prob4)
np.save('Experiment4/data.npy',data)
np.save('Experiment4/TrueHidden.npy',Sampling.hidden_data)



# Read the results
post_A=np.load('Post_A.npy')
post_B=np.load('Post_B.npy')
latent_seq=np.load('latent_seq.npy')
log_prob=np.loadtxt('log_prob.txt')
data=np.load('data.npy')
TrueHidden=np.load('TrueHidden.npy')
