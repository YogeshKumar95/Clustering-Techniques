#!/usr/bin/env python
# coding: utf-8

# In[374]:


import numpy as np
import sys
import math
import random
import collections
from numpy.random import seed
seed(1)

if len(sys.argv) != 5:
    sys.exit() 
X = np.genfromtxt(sys.argv[1],delimiter=",")
m = X.shape[0]
prev_labels = np.zeros(m)


# In[376]:


last_col = X.shape[1] 


# In[377]:


k = int(sys.argv[2])


# In[378]:


r = int(sys.argv[3])
counter = 0


# In[379]:


Overall_Q = sys.maxsize 
while(counter < r):
    Centroids = {}
    for i in range(k):
        Centroids[i+1] = X[random.randint(1, m)]
        
    #Running the iterations for Convergence
    while True: 
    #Assign the labels based on distance to each centroids 
        Labels = np.zeros(m)
        for i in range(m):
            Min_Distance = sys.maxsize
            for c in range(1,len(Centroids.keys())+1):
                temp = X[i] - Centroids.get(c)
                temp_dist = np.sum(temp**2)
                if (temp_dist <= Min_Distance):
                    Min_Distance = temp_dist
                    Labels[i] = c 
        if(collections.Counter(Labels) == collections.Counter(prev_labels)): 
            break;
        else:       
            Temp_Labels = np.reshape(Labels,(m,1))
            Overall = np.concatenate((X,Temp_Labels),axis=1)  
            last_col = Overall.shape[1] - 1
        
            #Compute the Quantization Error based on the Clusters
            Overall_Error = 0
            for i in range(m):
                Sum_d = 0
                val = Overall[i][last_col]
                Computation = Centroids.get(val)
                for j in range(last_col):
                    Sum_d = Sum_d + (Computation[j] - Overall[i][j])**2
                Overall_Error = Overall_Error + Sum_d
        
            #Compute the New Centroids
            Unique_vals,count_vals = np.unique(Overall[:,last_col], return_counts=True)
            Centroids = {}
            for i in range(len(Unique_vals)):
                Dat_subset = Overall[Overall[:,last_col]==Unique_vals[i]]
                Mean_subset = np.mean(Dat_subset[:,0:last_col],axis = 0)
                Centroids[Unique_vals[i]] = Mean_subset  
            prev_labels = Labels
    if(Overall_Error < Overall_Q):
        Overall_Q = Overall_Error
        final_labels = Labels
    counter = counter + 1
print(Overall_Q)
np.savetxt(sys.argv[4], final_labels, delimiter =',')


# In[ ]:





# In[ ]:




