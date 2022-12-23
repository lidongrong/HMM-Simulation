# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:04:39 2022

@author: s1155151972
"""

import matplotlib.pyplot as plt
import SD_generator as sdg
import Model as Model
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import os
import argparse
import utils
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# file paths
# Simulation Part
rate=[0.5,0.6,0.8,0.9,0.95,0.97]
paths=['D:\Object\PROJECTS\HMM\SynData\Rate0.5\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.6\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.8\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.9\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.95\PartialData',
      'D:\Object\PROJECTS\HMM\SynData\Rate0.97\PartialData']
# path of parameter
param_path='D:\Object\PROJECTS\HMM\SynData'
# names of parameter
param_names=['initial','transition','beta','mu','sigma']
# folder that stores results
parent_name='SimulationResults'
rate_name=[f'Rate{r}' for r in rate]
save_paths=utils.make_category(parent_name,rate_name)

# covariates
covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide']) 
types=np.array(['numeric','dummy'])
covariate_types=np.array(['numeric']*len(covariates))

# scale
sample_size=30000
iterations=40000
prob=0.5
nums=4
#data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types) 

# add arguments related to the optimizer
parser=argparse.ArgumentParser(description="tunning paramters")
parser.add_argument("-batch","--batch-size",default=40,type=int)
parser.add_argument("-lr","--learning-rate",default=0.01,type=float)
parser.add_argument("-hk","--hk",default=1,type=float)
parser.add_argument("-lbatch","--latent-batch-size",default=1000,type=int)
parser.add_argument("-SGLD","--use-sgld",default=True,type=bool)
parser.add_argument("-core","--num-core",default=0,type=int)
args=parser.parse_args(args=[])

# switchs related to optimizer
args.batch_size=50
args.hk=1
args.latent_batch_size=100
args.use_sgld=False
args.learning_rate=0.0005
args.num_core=0



if __name__ == '__main__':
    opts=utils.simulation(paths=paths,param_path=param_path,param_names=param_names,sample_size=sample_size,
                          covariates=covariates,covariate_types=covariate_types,args=args,
                          iterations=iterations,nums=nums,prob=prob)
    
    #utils.generate_graph(opts=opts,paths=paths,param_path=param_path)
    
    utils.pickle_data(opts=opts,save_paths=save_paths)


