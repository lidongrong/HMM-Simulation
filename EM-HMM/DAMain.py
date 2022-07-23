# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:28:08 2022

@author: lidon
"""


import os
import matplotlib.pyplot as plt
import math
import numpy as np
from ZMARGibbs import*
from multiprocessing import Pool
import multiprocessing as mp
from EMHMM import*
from DATool import*


#transition=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])
transition=np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.9,0.05,0.05],
                   [0.1,0.7,0.2],
                   [0.15,0.05,0.8]
    ])

#pi=np.array([0.7,0.2,0.1])
pi=np.array([0.6,0.3,0.1])

# different levels of missingness
rate=np.array([0.1,0.2,0.3,0.5,0.7,0.9])

sim_path='Result2022_7_21_22_55_46/SimulationResult'
naive_path='Result2022_7_21_22_55_46/NaiveResult'
comp_path='Result2022_7_21_22_55_46/CompleteResult'

num=30
sim_out=read_all(sim_path,rate,num)
naive_out=read_all(naive_path,rate,num)
comp_out=read_all(comp_path,rate,num)

sim_out=permute_all(sim_out,pi,transition,obs_prob,rate)
naive_out=permute_all(naive_out,pi,transition,obs_prob,rate)
comp_out=permute_all(comp_out,pi,transition,obs_prob,rate)

sim_out=np.array(sim_out)
naive_out=np.array(naive_out)
comp_out=np.array(comp_out)

sim_mae_pi,sim_mae_A,sim_mae_B=MAE(sim_out,pi,transition,obs_prob)
naive_mae_pi,naive_mae_A,naive_mae_B=MAE(naive_out,pi,transition,obs_prob)
comp_mae_pi,comp_mae_A,comp_mae_B=MAE(comp_out,pi,transition,obs_prob)