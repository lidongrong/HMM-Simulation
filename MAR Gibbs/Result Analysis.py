# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:55:01 2021

@author: s1155151972
"""


# The result analysis script is desinged to analyze all results in one-shot



import os

for i in range(0,8):
    os.mkdir(f'Experiment{i}')
    # Save the results
    np.save(f'Experiment{i}/Post_A.npy',out_obj[i].post_A)
    np.save(f'Experiment{i}/Post_B.npy',out_obj[i].post_B)
    np.save(f'Experiment{i}/latent_seq.npy',out_obj[i].latent_seq)
    np.savetxt(f'Experiment{i}/log_prob.txt',out_obj[i].log_prob)
    np.save(f'Experiment{i}/data.npy',out_obj[i].data)
    np.save(f'Experiment{i}/TrueHidden.npy',out_obj[i].true_hidden)