# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:03:22 2022

@author: lidon
"""

import SD_generator as sdg
import Model as Model
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import argparse
import os
import utils

rate=[0,0.5,0.6,0.8,0.9]

path=['D:\Object\PROJECTS\HMM\SynData\Rate0.5\FullData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.5\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.6\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.8\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.9\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.95\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.97\PartialData']
param_path='D:\Object\PROJECTS\HMM\SynData'
parent='D:\Object\PROJECTS\HMM\SimulationResult'
save_path=[f'rate{rate[i]}' for i in range(len(rate))]




covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide']) 
types=np.array(['numeric','dummy'])
d=3
covariates=covariates[0:d]
covariate_types=np.array(['numeric']*len(covariates))


sample_size=10000
num=4
iteration=80000
prob=0.6
log_step=5000
latent_batch_size=2500
num_core=8

#data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types)


if __name__ == '__main__':
    # define arg parser
    parser=argparse.ArgumentParser(description="tunning paramters")
    parser.add_argument("-batch","--batch-size",default=40,type=int)
    parser.add_argument("-lr","--learning-rate",default=0.01,type=float)
    parser.add_argument("-hk","--hk",default=1,type=float)
    parser.add_argument("-lbatch","--latent-batch-size",default=1000,type=int)
    parser.add_argument("-SGLD","--use-sgld",default=True,type=bool)
    parser.add_argument("-core","--num-core",default=0,type=int)
    args=parser.parse_args(args=[])
    
    # adjust argparser
    args.batch_size=sample_size
    args.hk=1
    args.latent_batch_size=latent_batch_size
    args.use_sgld=False
    # learning rate suggested by Wainwright, have a try
    args.learning_rate = (1/(126))*(1/(0.15*sample_size))
    args.num_core=num_core
    
    opts=[[] for i in range(len(path))]
    utils.make_category(parent,save_path)
    
    for i in range(len(path)):
        for j in range(num):
            data,lengths=sdg.data_loader(path[i],sample_size,covariates,covariate_types)
            m=Model.HMM_Model(data,lengths,covariates,covariate_types)
            optimizer=Model.Random_Gibbs(model=m,args=args,initial=None)
            param=optimizer.run(n=iteration,log_step=log_step,prog_bar=True,prob=prob,initial_x=None,initial_z=None)
            opts[i].append(optimizer)
            os.mkdir(f'{parent}\{save_path[i]}\Simulation{j}')
            optimizer.pickle(f'{parent}\{save_path[i]}\Simulation{j}',param_path)
    
  