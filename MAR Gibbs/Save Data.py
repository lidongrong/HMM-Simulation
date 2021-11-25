# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:56:06 2021

@author: s1155151972
"""

# Save the results
np.save('Post_A.npy',post_A)
np.save('Post_B.npy',post_B)
np.save('latent_seq.npy',latent_seq)
np.savetxt('log_prob.txt',log_prob)

# Read the results
post_A=np.load('Post_A.npy')
post_B=np.load('Post_B.npy')
latent_seq=np.load('latent_seq.npy')
log_prob=np.loadtxt('log_prob.txt')