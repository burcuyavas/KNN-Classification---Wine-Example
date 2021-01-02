#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
wine = datasets.load_wine()


# In[2]:


print(wine.feature_names)


# In[3]:


print(wine.target_names)


# In[4]:


print(wine.data[0:5])


# In[5]:


print(wine.target)


# In[7]:


print(wine.data.shape)
print(wine.target.shape)


# In[10]:


# Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)


# In[11]:


# Generating Model for K=5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[12]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




