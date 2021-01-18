#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import random
import sys
import collections
from numpy.random import seed
seed(1)
if len(sys.argv) != 5:
    sys.exit() 
X = np.genfromtxt(sys.argv[1],delimiter=",")
m = X.shape[0]
Distance = np.zeros((m,m))
for i in range(0,m):
    for j in range(0,m):
        Distance[i][j] = np.sum((X[i] - X[j])**2)


# In[212]:


Sigma = float(sys.argv[3])
k = int(sys.argv[2])


# In[213]:


Gammas = np.zeros((m,m))
for i in range(0,m):
    for j in range(0,m):
        Gammas[i][j] = Distance[i][j]/(2*(Sigma**2))


# In[216]:


weights = np.exp(-1*Gammas) 


# In[218]:


Diagonal = np.sum(weights, axis=1) 


# In[220]:


Diagonal_Matrix = np.diag(Diagonal)


# In[222]:


Laplace = Diagonal_Matrix - weights


# In[224]:


Normalized_Laplace = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        if ((i == j) and (Diagonal[i] != 0)):
            Normalized_Laplace[i][j] = 1
        elif ((i != j) and (Diagonal[i]!= 0) and (Diagonal[j]!=0)):
            Normalized_Laplace[i][j] = -weights[i][j]/math.sqrt(Diagonal[i]*Diagonal[j])
        else:
            Normalized_Laplace[i][j] = 0


# In[226]:


evals, evecs = np.linalg.eigh(Normalized_Laplace)


# In[228]:


Min_evecs = evecs[:,:k]


# In[230]:


U = Min_evecs.T


# In[232]:


Generalized_Vectors = np.zeros((k,m))
for i in range(k):
    for j in range(m):
        Generalized_Vectors[i][j] = U[i][j]/Diagonal[j]


# In[233]:


K_Means = Generalized_Vectors.T


# In[234]:


prev_labels = np.zeros((m))


# In[235]:


iterations = 0
r = 5
Overall_Q = sys.maxsize
while(iterations < r):
    Centroids = {}
    counter = 1
    Centroids[counter] = K_Means[random.randint(1, m)]

    #Compute the Centroids
    while(counter < k):
        Max_Dist = -sys.maxsize - 1
        for i in range(m):
            val_dist = 0
            Sum_dist = 0
            for key in Centroids:
                val = K_Means[i] - Centroids.get(key)
                val_dist = np.sum(val**2)
                Sum_dist = Sum_dist + val_dist
            if(Sum_dist >= Max_Dist):
                Max_Dist = Sum_dist
                Coordinates = K_Means[i]
        counter = counter + 1
        Centroids[counter] = Coordinates
     
    #Running the iterations for Convergence
    while True:
    #Assign the labels based on distance to each centroids 
        Labels = np.zeros(m)
        for i in range(m):
            Min_Distance = sys.maxsize
            for c in range(1,len(Centroids.keys())+1):
                temp = K_Means[i] - Centroids.get(c)
                temp_dist = np.sum(temp**2)
                if (temp_dist <= Min_Distance):
                    Min_Distance = temp_dist
                    Labels[i] = c
        if(collections.Counter(Labels) == collections.Counter(prev_labels)):
            break;
        else:
            Temp_Labels = np.reshape(Labels,(m,1))
            Overall = np.concatenate((K_Means,Temp_Labels),axis=1)
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
    iterations = iterations + 1
print(Overall_Q)
np.savetxt(sys.argv[4],Labels, delimiter =',')


# In[ ]:




