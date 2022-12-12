# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:42:18 2022

@author: lidon
"""

import SD_generator as sdg
import Model as Model
import numpy as np
import pandas as pd

path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\SynData\Rate0.9\PartialData'
covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide']) 
types=np.array(['numeric','dummy'])
covariate_types=np.array(['numeric']*len(covariates))
sample_size=50
data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types)  


m=Model.hmm_model(data,lengths,covariates,covariate_types)
opt=Model.gibbs(m)