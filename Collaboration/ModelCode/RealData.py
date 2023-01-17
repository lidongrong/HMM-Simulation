# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:52:05 2023

@author: lidon
"""

import pandas as pd
import os
import SD_generator as sdg
import Model
import matplotlib.pyplot as plt
import Model as Model
import numpy as np
import multiprocessing as mp
import time
import os
import argparse
import torch
from RealUtils import*
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data_path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\ETVTDF timeseries\ETVTDF timeseries\DesignMatrix3'

covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide','Sex','Age'])
covariate_types=np.array(['numeric']*len(covariates))

sample_size=1000

# read data
sequences=data_reader(data_path,sample_size)
lengths=data_to_length(sequences)
sequences=data_to_tensor(sequences,stages)
# convert to numpy
lengths=lengths.numpy()
data=sequences.numpy()
# drop empty sequences
l=lengths[lengths!=0]
data=data[lengths!=0]
lengths=l

if __name__ == '__main__':
    
    m=Model.HMM_Model(data,lengths,covariates,covariate_types)
    x,y=m.split()
    
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
    args.latent_batch_size=50
    args.use_sgld=False
    # learning rate suggested by Wainwright, have a try
    args.learning_rate = (1/126)*(1/(0.25*sample_size))
    args.num_core=0
    
    
    optimizer=Model.Random_Gibbs(model=m,args=args,initial=None)
    
    '''
    zt=optimizer.sample_zt(x[0],y[0],z[0],optimizer.model.lengths[0],
                        optimizer.model.x_masks[0],optimizer.model.y_masks[0])
    '''
    
    param=optimizer.run(n=280,log_step=100,prog_bar=True,prob=1,initial_x=None,initial_z=None)
