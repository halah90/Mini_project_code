#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.linear_model import LogisticRegression


# In[4]:


st.write("""
# This is Blood Donation Prediction App
This app predicts the **Blood Donation** for Future Expectancy!
""")


# In[5]:


st.sidebar.header('User Input Parameters')
def user_input_features():
    Recency = st.sidebar.slider('Recency(months)', 0, 30, 21)
    Frequency = st.sidebar.slider('Frequency(times)',0, 60, 1)
    Monetary  = st.sidebar.slider('Monetary (c.c. blood)', 100, 15000, 250)
    Time = st.sidebar.slider('Time (months)', 1, 100, 21)
    data = {'Recency (months)': Recency,
            'Frequency (times)': Frequency,
            'Monetary (c.c. blood)': Monetary,
            'Time (months)': Time}
    feature = pd.DataFrame(data, index=[0])
    return feature

df = user_input_features()


# In[6]:


st.subheader('User Input parameters')
st.write(df)


# In[7]:


#prepare data 
tpot_data = pd.read_csv("C:/Users/user/Desktop/halah_work/Technolabs/mini_project/transfusion.data")
tpot_data.columns=['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)',
       'Time (months)', 'target']


# In[8]:


features = tpot_data.drop('target', axis=1)
#features["Monetary (c.c. blood)"]=np.log(features["Monetary (c.c. blood)"])
#features["Time (months)"]=np.log(features["Time (months)"])
y=tpot_data['target']


# In[9]:


logReg = LogisticRegression(random_state=0)
logReg.fit(features,y)
#np.log(df["Monetary (c.c. blood)"])
predictions=logReg.predict(df)


# In[10]:


st.subheader('Prediction')
st.write(predictions[0])

