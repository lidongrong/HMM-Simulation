# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:04:56 2022

@author: lidon
"""

from SD_generator import*

# covariates
variates={'HBV': np.array(['HBeAg+ALT<=1ULN', 'HBeAg+ALT>1ULN', 'HBeAg-ALT<=1ULN',
        'HBeAg-ALT>1ULN','Cirr','HCC','Death'], dtype=object),
 'Lab': np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
        'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
        'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC'],
       dtype=object),
 'Med': np.array(['ACEI', 'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
        'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
        'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
        'Tenofovir Disoproxil Fumarate', 'Thiazide'], dtype=object)}
covariates=np.array(list(variates['Lab'])+list(variates['Med']))

d=3
covariates=covariates[0:d]
d=len(covariates)
initial=np.array([0.28571429, 0.23809524, 0.19047619, 0.14285714, 0.0952381, 0.04761905, 0.0])
transition=np.array([[0.6,0.1,0.1,0.1,0.05,0.05,0],
                     [0.1,0.6,0.1,0.1,0.05,0.05,0],
                     [0.1,0.1,0.6,0.1,0.05,0.05,0],
                     [0.1,0.1,0.1,0.6,0.05,0.05,0],
                     [0,0,0,0.05,0.8,0.1,0.05],
                     [0,0,0,0,0.1,0.7,0.2],
                     [0,0,0,0,0,0,1]])
# beta[z]=0.3*z for each coordinator
#beta=[[np.random.multivariate_normal(np.zeros(d),0.01*np.eye(d),1)[0] for i in range(len(initial)-1)] for j in range(len(initial))]
beta=np.random.uniform(0,2,(len(initial),len(initial)-1,d))
beta=((-1)**np.random.binomial(1,0.5,beta.shape))*beta
# mu[z]=-3+z for each coordinator
mu=np.array([np.random.multivariate_normal(np.zeros(d),1*np.eye(d),1)[0] for i in range(len(initial))])
# sigma set to be identity
sigma=np.array([np.eye(len(covariates)) for k in range(len(initial))])

params=[initial,transition,beta,mu,sigma]
params_name=['initial','transition','beta','mu','sigma']

# total sequence number
num=40000
# missing rate
rate=[0.5,0.6,0.8,0.9,0.95,0.97]
# path
path='D:\Object\PROJECTS\HMM\SynData'


if __name__=='__main__':
    synthesizer=Synthesize(covariates,initial,transition,beta,mu,sigma,num,lbd=20)
    full=synthesizer.generate_sequences()
    #save params
    for i in range(len(params)):
        np.save(f'{path}/{params_name[i]}.npy',params[i])
    for r in range(len(rate)):
        tmp_path=f'{path}/Rate{rate[r]}'
        os.mkdir(tmp_path)
        partial=synthesizer.generate_partial_sequences(rate[r])
        synthesizer.save_data(tmp_path)
    
    
