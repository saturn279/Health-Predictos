#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('cancer.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[6]:


import seaborn as sns


# In[8]:


sns.set_style('whitegrid')
sns.countplot(x = 'diagnosis', data = data)


# In[11]:


dataset = data
dataset['diagnosis'].replace(['M','B'], [1,0], inplace = True)


# In[15]:


dataset.drop('Unnamed: 32',axis = 1, inplace = True)


# In[16]:


corr = dataset.corr()
plt.figure(figsize = (25,25))
sns.heatmap(corr, annot = True)


# In[17]:


dataset.corr()


# In[18]:


dataset.drop(['id','symmetry_se','smoothness_se','texture_se','fractal_dimension_mean'], axis = 1, inplace = True)


# In[19]:


dataset.head()


# In[21]:


plt.figure(figsize = (25,25))
sns.heatmap(dataset.corr(), annot = True)


# In[22]:


X = dataset.drop('diagnosis', axis = 1)
y = dataset['diagnosis']


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[25]:


print("Train Set: ", X_train.shape, y_train.shape)
print("Test Set: ", X_test.shape, y_test.shape)


# In[26]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)


# In[31]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[28]:


confusion_matrix(y_test, model.predict(X_test))


# In[32]:


print(f"Accuracy is {round(accuracy_score(y_test, model.predict(X_test))*100,2)}")


# ## Applying Hyperparameter Tuning

# In[33]:


from sklearn.model_selection import RandomizedSearchCV


# In[34]:


classifier = RandomForestClassifier(n_jobs = -1)


# In[35]:


from scipy.stats import randint
param_dist={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,300,400,500],
              'max_features':randint(1,27),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,27),
              }


# In[36]:


search_clfr = RandomizedSearchCV(classifier, param_distributions = param_dist, n_jobs=-1, n_iter = 40, cv = 9)


# In[37]:


search_clfr.fit(X_train, y_train)


# In[38]:


params = search_clfr.best_params_
score = search_clfr.best_score_
print(params)
print(score)


# In[45]:


claasifier=RandomForestClassifier(n_jobs=-1, n_estimators=200,bootstrap= True,criterion='gini',max_depth=20,max_features=8,min_samples_leaf= 1)


# In[46]:


classifier.fit(X_train, y_train)


# In[47]:


confusion_matrix(y_test, classifier.predict(X_test))


# In[48]:


print(f"Accuracy is {round(accuracy_score(y_test, classifier.predict(X_test))*100,2)}%")


# In[49]:


import pickle
pickle.dump(classifier, open('cancer.pkl', 'wb'))

