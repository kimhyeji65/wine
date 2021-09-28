#!/usr/bin/env python
# coding: utf-8

# # (1) 필요한 모듈 import하기

# In[45]:


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score


# # (2) 데이터 준비

# In[6]:


wine = load_wine()
wine.keys()


# In[23]:



wine_feature = wine.data
wine.data.shape


# In[19]:


wine_label = wine.target
wine.target


# In[29]:


wine.target_names


# In[31]:


wine.DESCR


# # (3) train, test 데이터 분리

# In[33]:


wine_X_train, wine_X_test, wine_y_train, wine_y_test = train_test_split(wine_feature, wine_label, test_size=0.2, random_state = 7)
                                                   
                            


# #  (4) 모델 학습 및 예측
# 

# In[59]:


#decision
decision_tree = DecisionTreeClassifier(random_state=7)
decision_tree.fit(wine_X_train, wine_y_train)
y_pred = decision_tree.predict(wine_X_test)
print(classification_report(wine_y_test, y_pred))


# In[58]:


# randomforest
random_forest = RandomForestClassifier(random_state=32)
random_forest.fit( wine_X_train, wine_y_train)
y_pred = random_forest.predict(wine_X_test)
print(classification_report(wine_y_test, y_pred))


# In[61]:


#SVM
svm_model = svm.SVC()
svm_model.fit(wine_X_train, wine_y_train)
y_pred = svm_model.predict(wine_X_test)

print(classification_report(wine_y_test, y_pred))


# In[63]:


#SGD Classifier
sgd_model = SGDClassifier()
sgd_model.fit(wine_X_train, wine_y_train)

y_pred = sgd_model.predict(wine_X_test)

print(classification_report(wine_y_test, y_pred))


# In[66]:


#LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(wine_X_train, wine_y_train)
y_pred = logistic_model.predict(wine_X_test)

print(classification_report(wine_y_test, y_pred))


# In[ ]:




