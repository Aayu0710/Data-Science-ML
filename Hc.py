#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Hiercial clustering


# In[21]:


#import libaries
import pandas as pd
import matplotlib.pyplot as plt
#import data set
dataset = pd.read_csv("C:\\Users\\HP\\Documents\\Mall.csv")
x = dataset.iloc[:, :].values
x


# In[23]:


# wants to decide how many clusters we want
#dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customer")
plt.ylabel("Euclidian distance")
plt.show()


# In[27]:


# Train the model
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 2)
y_hc = clustering.fit_predict(x)
y_hc


# In[29]:


#visualising the clusters
plt.scatter(x[y_hc==0, 0], x[y_hc==0, 1], c = 'red', label = 'Cluster-1')
plt.scatter(x[y_hc==1, 0], x[y_hc==1, 1], c = 'green', label = 'Cluster-2')
plt.title('Cluster of customers')
plt.xlabel("Anual income(k$)")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()


# In[ ]:




