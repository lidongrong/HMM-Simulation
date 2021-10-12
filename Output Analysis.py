# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:01:02 2021

@author: s1155151972
"""


import matplotlib.pyplot as plt

# draw posterior of a11
dist=[]

for x in post_A:
    dist.append(x[0][0])

plt.xlabel(f"a11 with missing rate p=0.4 \n 95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})")
plt.ylabel("frequency")
plt.text(0.38,0,f'95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})')
plt.hist(dist,bins=60)


dist=[]
for x in post_A:
    dist.append(x[1][1])

plt.xlabel(f"a22 with missing rate p=0.4 \n 95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})")
plt.ylabel("frequency")

plt.hist(dist,bins=60)

dist=[]
for x in post_A:
    dist.append(x[2][2])

plt.xlabel(f"a33 with missing rate p=0.4 \n 95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})")
plt.ylabel("frequency")
#plt.text(0.55,0,f'95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})')
plt.hist(dist,bins=60)

dist=[]
for x in post_A:
    dist.append(x[3][3])

plt.xlabel(f"a44 with missing rate p=0.4 \n 95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})")
plt.ylabel("frequency")
#plt.text(0.44,0,f'95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})')
plt.hist(dist,bins=60)

dist=[]
for x in post_B:
    dist.append(x[0][0])

plt.xlabel(f"b11 with missing rate p=0.4 \n 95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})")
plt.ylabel("frequency")
#plt.text(0.57,0,f'95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})')
plt.hist(dist,bins=60)

dist=[]
for x in post_B:
    dist.append(x[1][3])

plt.xlabel(f"b24 with missing rate p=0.4 \n 95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})")
plt.ylabel("frequency")
#plt.text(0.3,0,f'95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})')
plt.hist(dist,bins=60)

dist=[]
for x in post_B:
    dist.append(x[2][4])

plt.xlabel(f"b35 with missing rate p=0.4 \n 95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})")
plt.ylabel("frequency")
#plt.text(0,0,f'95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})')
plt.hist(dist,bins=60)

dist=[]
for x in post_B:
    dist.append(x[4][4])

plt.xlabel(f"b55 with missing rate p=0.4 \n 95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})")
plt.ylabel("frequency")
#plt.text(0.62,0,f'95% percentile:\n ({np.percentile(dist,2.5)},\n {np.percentile(dist,97.5)})')
plt.hist(dist,bins=60)
