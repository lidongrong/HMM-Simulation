# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

Px=pd.read_parquet('Px.parquet')
Dx=pd.read_parquet('Dx.parquet')
Lab=pd.read_parquet('Lab.parquet')
Med=pd.read_parquet('Med.parquet')

Px.to_csv('Px.csv')
Dx.to_csv('Dx.csv')
Lab.to_csv('Lab.csv')
Med.to_csv('Med.csv')
