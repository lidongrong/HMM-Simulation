# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:08:50 2022

@author: lidon
"""

from patient import*
import os


variates={'HBV': np.array(['HBeAg+ALT<=1ULN', 'HBeAg+ALT>1ULN', 'HBeAg-ALT<=1ULN',
        'HBeAg-ALT>1ULN'], dtype=object),
 'Lab': np.array(['ACEI', 'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
        'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
        'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
        'Tenofovir Disoproxil Fumarate', 'Thiazide'], dtype=object),
 'Med': np.array(['ACEI', 'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
        'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
        'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
        'Tenofovir Disoproxil Fumarate', 'Thiazide'], dtype=object)}


pt=[]
os.mkdir(f'{path}/DesignMatrix')
new_path=f'{path}/DesignMatrix'
strange=[]
strange=pd.Series(strange)
for i in range(0,len(pid)):
    print(i)
    p=patient(pid[i])
    p.align_date() 
    if ('Result' in p.ts.columns):
        p.construct_matrix()
        p.data.to_csv(f'{new_path}/{pid[i]}.csv')
    else:
        strange[len(strange)]=i
        strange.to_csv(f'{new_path}/NoResultPatient.txt')

strange=pd.DataFrame(strange)
strange.to_csv(f'{new_path}/NoResultPatient.txt')
    
    