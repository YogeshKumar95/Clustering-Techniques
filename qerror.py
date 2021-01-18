#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
import sys


# In[ ]:

#Validation on Terminal Command
if len(sys.argv) != 3:
    sys.exit() 


# In[ ]:

#Reading from the Terminal
X = np.genfromtxt(sys.argv[1],delimiter=",")
Y = np.genfromtxt(sys.argv[2],delimiter=",")


# In[ ]:


m = Y.shape[0]
Y = np.reshape(Y,(m,1))
Overall = np.concatenate((X,Y),axis=1)


# In[ ]:


last_col = Overall.shape[1] - 1
Unique_vals,count_vals = np.unique(Overall[:,last_col], return_counts=True)


# In[ ]:

#Compute the Centroids of the Original Data
Centroids = {}
for i in range(len(Unique_vals)):
    Dat_subset = Overall[Overall[:,last_col]==Unique_vals[i]]
    Mean_subset = np.mean(Dat_subset[:,0:last_col],axis = 0)
    Centroids[Unique_vals[i]] = Mean_subset


# In[ ]:

#Print the Overall Quantization Error
Overall_Error = 0
for i in range(m):
    Sum_d = 0
    val = Overall[i][last_col]
    Computation = Centroids.get(val)
    for j in range(last_col):
        Sum_d = Sum_d + (Computation[j] - Overall[i][j])**2
    Overall_Error = Overall_Error + Sum_d


# In[ ]:


print(Overall_Error)

