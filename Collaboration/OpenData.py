# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os

# Store the path of the data
path='D:\Files\CUHK Material\Research_MakeThisObservable\EMR\Data'

# Read data and return them as pandas Dataframe
def data_reader(path):
    # enter the path where you store your data in
    os.chdir(path)
    Px=pd.read_parquet('Px.parquet')
    Dx=pd.read_parquet('Dx.parquet')
    Lab=pd.read_parquet('Lab.parquet')
    Med=pd.read_parquet('Med.parquet')
    return Px,Dx,Lab,Med

# Save data as .csv files
def safe_data():
    Px.to_csv('Px.csv')
    Dx.to_csv('Dx.csv')
    Lab.to_csv('Lab.csv')
    Med.to_csv('Med.csv')