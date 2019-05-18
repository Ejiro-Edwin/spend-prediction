
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score 


# In[4]:


test = pd.read_csv("test_technidus.csv")
train = pd.read_csv("train_technidus.csv")
train['BirthDate']=pd.to_datetime(train['BirthDate'])
test['BirthDate']=pd.to_datetime(test['BirthDate'])


# In[5]:


train['BirthYear'] = train['BirthDate'].dt.year
test['BirthYear'] = test['BirthDate'].dt.year
train['Age'] = 2019 - train['BirthYear']
test['Age'] = 2019 - test['BirthYear']


# In[6]:


pd.isnull(train).any()


# In[7]:


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# In[8]:


for col in ['CountryRegionName', 'Education', 
          'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 
          'NumberCarsOwned','TotalChildren', 'BikeBuyer']:
    train=create_dummies(train, col)
    test=create_dummies(test, col)


# In[9]:


columns= ['CustomerID', 'YearlyIncome', 'Age', 
          'CountryRegionName_Australia', 'CountryRegionName_Canada',
          'CountryRegionName_France', 'CountryRegionName_Germany',
          'CountryRegionName_United Kingdom', 'CountryRegionName_United States',
          'Education_Bachelors ', 'Education_Graduate Degree',
          'Education_High School', 'Education_Partial College',
          'Education_Partial High School', 'Occupation_Clerical',
          'Occupation_Management', 'Occupation_Manual', 'Occupation_Professional',
          'Occupation_Skilled Manual', 'Gender_F', 'Gender_M', 'MaritalStatus_M',
          'MaritalStatus_S', 'HomeOwnerFlag_0', 'HomeOwnerFlag_1',
          'NumberCarsOwned_0', 'NumberCarsOwned_1', 'NumberCarsOwned_2',
          'NumberCarsOwned_3', 'NumberCarsOwned_4', 'TotalChildren_0',
          'TotalChildren_1', 'TotalChildren_2', 'TotalChildren_3',
          'TotalChildren_4', 'TotalChildren_5', 'BikeBuyer_0', 'BikeBuyer_1']


# In[10]:


columns = train.columns

