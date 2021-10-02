# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 20:50:34 2021

@author: a
"""

b1=[]

for x in b:
    b1.append(x[0][2])

import matplotlib.pyplot as plt
plt.hist(b1,bins=50)
plt.show()