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
os.mkdir('Experiment5')





# Save the results
np.save('Experiment1/Post_A.npy',post_A)
np.save('Experiment1/Post_B.npy',post_B)
np.save('Experiment1/latent_seq.npy',latent_seq)
np.savetxt('Experiment1/log_prob.txt',log_prob)
np.save('Experiment1/data.npy',data)
np.save('Experiment1/TrueHidden.npy',Sampling.hidden_data)


np.save('Experiment2/Post_A1.npy',post_A)
np.save('Experiment2/Post_B1.npy',post_B)
np.save('Experiment2/latent_seq1.npy',latent_seq)
np.savetxt('Experiment2/log_prob1.txt',log_prob)
np.save('Experiment2/data.npy',data)
np.save('Experiment2/TrueHidden.npy',Sampling.hidden_data)

np.save('Experiment3/Post_A2.npy',post_A)
np.save('Experiment3/Post_B2.npy',post_B)
np.save('Experiment3/latent_seq2.npy',latent_seq)
np.savetxt('Experiment3/log_prob2.txt',log_prob)
np.save('Experiment3/data.npy',data)
np.save('Experiment3/TrueHidden.npy',Sampling.hidden_data)

np.save('Experiment4/Post_A3.npy',post_A)
np.save('Experiment4/Post_B3.npy',post_B)
np.save('Experiment4/latent_seq3.npy',latent_seq)
np.savetxt('Experiment4/log_prob3.txt',log_prob)
np.save('Experiment4/data.npy',data)
np.save('Experiment4/TrueHidden.npy',Sampling.hidden_data)

np.save('Experiment5/Post_A4.npy',post_A)
np.save('Experiment5/Post_B4.npy',post_B)
np.save('Experiment5/latent_seq4.npy',latent_seq)
np.savetxt('Experiment5/log_prob4.txt',log_prob)
np.save('Experiment5/data.npy',data)
np.save('Experiment5/TrueHidden.npy',Sampling.hidden_data)



# Read the results
post_A=np.load('Post_A.npy')
post_B=np.load('Post_B.npy')
latent_seq=np.load('latent_seq.npy')
log_prob=np.loadtxt('log_prob.txt')
data=np.load('data.npy')
TrueHidden=np.load('TrueHidden.npy')
