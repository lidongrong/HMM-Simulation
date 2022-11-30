# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:31:46 2022

@author: s1155151972
"""

# import necessary packages
import logging
import os
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro

import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.infer
import pyro.infer.enum
from pyro.infer import config_enumerate
import pyro.poutine as poutine
from torch._C import NoneType

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# NECESSARY SETTINGS
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.2')

pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format='%(message)s', level=logging.INFO)

# Set matplotlib settings
#%matplotlib inline
plt.style.use('default')



# MODEL
#@title
from torch._C import NoneType
#@title
@config_enumerate
def hmm_model(sequences,lengths,args,batch_size=16):
  assert not torch._C._get_tracing_state()

  # num_sequences : total number of sequences
  # max_length: maximum of lengths
  # data_dim: dimension of all covariates+1 (for obs state)
  num_sequences, max_length, data_dim = sequences.shape
  covariate_dim=data_dim-args.hidden_dim

  # initial distributionprint('test')
  initial=pyro.sample(
      "initial",
      dist.Dirichlet(torch.ones(args.hidden_dim))
        )  
  # transition probability
  with pyro.plate('transition_components',args.hidden_dim):
    transition=pyro.sample(
        'transition',
        dist.Dirichlet(torch.ones(args.hidden_dim))
    )

  # distribution parameters for x_t|z_t 
  # assume mean mu with variance sigma^2
  with pyro.plate('mu_plate',args.hidden_dim):
    mu=pyro.sample(
        'mu',
        dist.MultivariateNormal(torch.zeros(covariate_dim),torch.eye(covariate_dim))
    )
  # currently not useful sigma 
  '''
  sigma=pyro.sample(
      "sigma",
      dist.HalfNormal(scale=1.0).expand([args.hidden_dim,data_dim-args.hidden_dim]).to_event(1)
  )
  '''
  # distribution parameters for y_t|x_t,z_t
  '''
  with pyro.plate('beta_components',args.hidden_dim):
    beta=pyro.sample(
        'beta',
        dist.Normal(0,1).expand([args.hidden_dim,covariate_dim]).to_event(1)
    )
    '''
  with pyro.plate('beta_plate1',args.hidden_dim):
    with pyro.plate('beta_plate2',args.hidden_dim):
      beta=pyro.sample(
          'beta',
          dist.MultivariateNormal(torch.zeros(covariate_dim),torch.eye(covariate_dim))
      )
  
  #y_emission=dist.Categorical(torch.exp(torch.matmul(beta,x)))
  #define the emission process within the markov chain
  tones_plate = pyro.plate("tones", args.hidden_dim, dim=-1)
  x_plate=pyro.plate('trunc_mu',args.hidden_dim)
  assert batch_size==args.batch_size
  for i in pyro.plate("sequences",len(sequences),batch_size):
    length=lengths[i]
    sequence=sequences[i,:length]
    # sample the first hidden state from initial
    z=0
    for t in pyro.markov(range(length)):
      if t==0:
        z=pyro.sample(
            "z_{}_{}".format(i,t),
            dist.Categorical(initial)
        )
      else:
        z=pyro.sample(
            "z_{}_{}".format(i,t),
            dist.Categorical(transition[z])
        )

      # emission process
      ob1=sequence[t][:-args.hidden_dim]
      is_observed=~ob1.isnan()
      is_observed=torch.tensor(is_observed).clone().detach()
      # handle missing
      valid_data=ob1.clone()
      valid_data[ob1.isnan()]=0
      # only model observed cases
      '''
      # method 1: directly sample (works)
      x=pyro.sample(f"x_{i}_{t}",
                dist.MultivariateNormal(mu[z],torch.eye(covariate_dim)),
                obs=valid_data)
      print('x shape:',x.shape)
      '''
      '''
      # method 2: by mask, fails
      with poutine.mask(mask=~is_observed):
        x=pyro.sample(
            f"x_{i}_{t}",
            dist.MultivariateNormal(mu[z],torch.eye(covariate_dim)),
            obs=valid_data
        )
      print('x shape: ',x.shape)
      '''
      '''
      # method 3: directly sample: works
      x=pyro.sample(
            f"x_{i}_{t}",
            dist.Normal(mu[z],torch.ones(covariate_dim)).independent(1),
            obs=valid_data
        )
      '''
      # method 4: hands on imputing
      
      x=ob1.clone()
      is_missing=ob1.isnan()
      missing_idx=torch.nonzero(is_missing)
      
      #print('mu shape:',mu.shape)
      for k in x_plate:
        trunc_mu=pyro.sample(
            f'trunc_mu_{i}_{t}_{k}',
            dist.MultivariateNormal(mu[k,is_missing],0.001*torch.eye(sum(is_missing)))
        )
      #print('trunc mu shape:',trunc_mu.shape)
      # in case missing

      if is_missing.any():
        x_impute=pyro.sample(
            f"x_impute_{i}_{t}",
            dist.MultivariateNormal(trunc_mu,torch.eye(sum(is_missing)))
        )
        #print('impute shape: ',x_impute.shape)
        x[is_missing]=x_impute
        #print('x shape:',x.shape)
      pyro.sample(
          f"x_{i}_{t}",
          dist.MultivariateNormal(mu[z],torch.eye(covariate_dim)),
          obs=x
      )
      
      # observational process
      # handle missingness
      weight=torch.exp(-torch.matmul(beta,valid_data))
      weight=weight/sum(weight)
      ob2=sequence[t][-args.hidden_dim:]
      is_observed=~ob2.isnan()
      is_observed=torch.tensor(is_observed).clone().detach()
      valid_data=ob2.clone()
      valid_data[ob2.isnan()]=0
      #print('z squeeze: ',z.squeeze(-1))
      if sum(valid_data)!=0:
        with tones_plate:
          y=pyro.sample(
              "y_{}_{}".format(i,t),
              #dist.MaskedDistribution(y_emission,~ob2.isnan()),
              dist.Bernoulli(weight[z.squeeze(-1)]),
              #dist.Bernoulli(weight[z.squeeze(-1)]),
              obs=valid_data
                )

      '''
      with poutine.mask(mask=~is_observed):
        x=pyro.sample(
          "x_{}_{}".format(i,t),
          dist.MultivariateNormal(mu[z],torch.eye(covariate_dim)),
          obs= valid_data
              )
      '''       
        
      '''
      # observation process
      # sample from multinomial with prob
      # exp(-beta*x)
      #weight=torch.exp(-torch.matmul(beta[z],x))
      weight=torch.exp(-torch.matmul(beta[z],valid_data))
      weight=weight/sum(weight)
      # y_emission=dist.Categorical(weight)
      # handle missingness
      ob2=sequence[t][-args.hidden_dim:]
      is_observed=~ob2.isnan()
      valid_data=ob2.clone()
      valid_data[~ob2.isnan()]=0

      print('start evaluating...')
      with poutine.mask(mask=is_observed):
        y=pyro.sample(
            "y_{}_{}".format(i,t),
            #dist.MaskedDistribution(y_emission,~ob2.isnan()),
            dist.Categorical(weight),
            #dist.Bernoulli(weight[z.squeeze(-1)]),
            obs=valid_data
              )
              '''
              
        
            
      
      
      
        



# DATA PREPROCESSOR

# Path that stores the data
data_path='DesignMatrix'
#os.listdir(data_path)

def data_reader(data_path,sample_size):
  f=os.listdir(data_path)
  f=f[0:sample_size]
  for i in range(0,len(f)):
    f[i]=data_path+'/'+f[i]
  sequences=[]
  for k in range(0,sample_size):
    sequences.append(pd.read_csv(f[k]))
  return sequences

# convert data to tensor
# convert sequences to tensor
def data_to_tensor(sequences,stages):
  # sequences: must be a list of pd frames
  # stages: the observed stage, array of strings
  # stage in stages will be turned into
  # int types
  
  # first, acquire the maximum lengths
  lengths=data_to_length(sequences)
  max_length=np.int32(max(lengths))

  #features=sequences[0].columns
  for i in range(0,len(sequences)):
    # drop stage column
    sequences[i]=sequences[i].drop(['Stage','Unnamed: 0','TimePoint'],axis=1)
    # standardize ages to avoid explosion
    sequences[i]['Age']=sequences[i]['Age']/100
    features=sequences[i].columns
    # standardize data
    for k in range(0,len(sequences[i].columns)-9):
      seq=sequences[i][sequences[i].columns[k]]
      seq=(seq-seq.mean())/seq.std()
      sequences[i][sequences[i].columns[k]]=seq

    sequences[i]=sequences[i].values
    # if sequence[i] shorter than max_length
    # impute them with nan
    # but these imputed nan will be ignored
    # in training by specifications of lengths
    # of each sequence
    if sequences[i].shape[0]<max_length:
      compensate=np.empty((max_length-sequences[i].shape[0],len(features)))
      compensate[:]=np.nan
      sequences[i]=np.concatenate((sequences[i],compensate),axis=0)
    
    for k in range(0,len(sequences[i])):
      for r in range(0,len(stages)):
        if sequences[i][k][len(sequences[i][k])-1]==stages[r]:
          sequences[i][k][len(sequences[i][k])-1]=r
          break
  sequences=np.float32(sequences)
  sequences=torch.tensor(sequences)
  return sequences


def data_to_length(sequences):
  # sequences is the list of data frames
  # returned by data_reader
  # return a tensor, recording the length
  # of each array
  lengths=[]
  for k in range(0,len(sequences)):
    lengths.append(sequences[k].values.shape[0])
  return torch.tensor(lengths)
  
 
# all observable stages
stages=['HBeAg+ALT<=1ULN', 'HBeAg+ALT>1ULN', 'HBeAg-ALT<=1ULN',
                       'HBeAg-ALT>1ULN','Cirr','HCC','Death']

# TEST function, turn nan values to 0
def data_impute(sequences,hidden_dim=7):
  #sequences=np.array(sequences)
  sequences[np.isnan(sequences)]=0
  # further impute observation
  for i in range(sequences.shape[0]):
    for j in range(sequences.shape[1]):
      site=np.random.randint(1,hidden_dim)
      sequences[i][j][-site]=1
  
  return sequences

# DATA PREPROCES
sequences=data_reader(data_path,3000)

lengths=data_to_length(sequences)
sequences=data_to_tensor(sequences,stages)


# ADD PARSER
# set hidden dim by default
parser=argparse.ArgumentParser(
    description="Argument for HMM Training"
)
# specify hidden dimension
parser.add_argument("-hd","--hidden-dim",default=7,type=int)
parser.add_argument("-b","--batch-size",default=8,type=int)
args=parser.parse_args(args=[])


# RUN THE CODE VIA SVI

## auto guider
#model_guide=pyro.infer.autoguide.AutoMultivariateNormal(hmm_model)
model_guide=pyro.infer.autoguide.AutoDelta(poutine.block(hmm_model,expose=['transition','initial','mu','beta']))
#model_guide=pyro.infer.autoguide.AutoMultivariateNormal(poutine.block(hmm_model,expose=['initial']))
#pyro.render_model(model_guide,model_args=(sequences,lengths,args))

pyro.clear_param_store()
pyro.set_rng_seed(1)
#optimizer
adam=pyro.optim.Adam({"lr":0.05})
elbo=pyro.infer.TraceEnum_ELBO()
guide=model_guide
svi=pyro.infer.SVI(hmm_model,model_guide,adam,elbo)

# training
losses=[]
for step in range(8000 if not smoke_test else 2):  # Consider running for more steps.
    loss = svi.step(sequences,lengths,args,batch_size=args.batch_size)
    losses.append(loss)
    if step % 1 == 0:
      print('iter:',step)
      logging.info("Elbo loss: {}".format(loss))

'''
our_model=hmm_model
kernel = pyro.infer.mcmc.NUTS(our_model, jit_compile=True)
#kernel=pyro.infer.mcmc.NUTS(our_model)
sampler=pyro.infer.mcmc.MCMC(kernel,num_samples=300,warmup_steps=100,num_chains=1,disable_progbar=False)
posterior=sampler.run(sequences,lengths,args)
'''