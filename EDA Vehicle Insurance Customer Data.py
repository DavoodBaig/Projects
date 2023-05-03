#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
table1=pd.read_csv("customer_details.csv")
table2=pd.read_csv("customer_policy_details.csv")


# In[3]:


table1


# In[5]:


table1.columns=['Customer_id','Gender','Age','driving_license_present','region code','previously insured','vehicle age','vehicle damage']


# In[6]:


table2


# In[7]:


table2.columns=['Customer_id','annual premium(in Rs)','sales channel code','vintage','response']


# In[8]:


table1=table1.set_index('Customer_id')


# In[9]:


table2.set_index('Customer_id')


# In[10]:


table1


# In[11]:


table2


# In[12]:


null1=pd.isnull(table1).sum()
print("The items in table1 that are null in each column are:\n",null1)


# In[13]:


null2=pd.isnull(table2).sum()
print("The items int table2 that are null in each column are:\n",null2)

#isnull will give us the boolean value as true or false on particular case.
# null will 


# In[ ]:





# In[14]:


table1.dropna(axis=1)


# In[15]:


table2.dropna(axis=1)


# In[ ]:





# In[16]:


table1=table1.fillna(0)


# In[17]:


table2=table2.fillna(0)


# In[18]:


table1


# In[20]:


table2


# In[19]:


table1["Age"].fillna(table1["Age"].mean())


# In[21]:


table1["region code"].fillna(table1["region code"].mean())


# In[22]:


table1["previously insured"].fillna(table1["previously insured"].mean())


# In[ ]:





# In[23]:


table2["annual premium(in Rs)"].fillna(table2["annual premium(in Rs)"].mean())


# In[24]:


table2["sales channel code"].fillna(table2["sales channel code"].mean())


# In[25]:


table2["vintage"].fillna(table2["vintage"].mean()) 


# In[26]:


table2["response"].fillna(table2["response"].mean())


# In[ ]:





# In[29]:


table1.mode()


# In[30]:


table2.mode()


# In[31]:


table1


# In[57]:


table2


# In[32]:


outliers1=table1.describe()[['Age','driving_license_present','region code']]
print(outliers1)


# In[33]:


outliers2=table2.describe()[['annual premium(in Rs)','sales channel code','vintage','response']]
print(outliers2)


# In[ ]:





# In[34]:


outliers1.mean()


# In[35]:


outliers2.mean()


# In[ ]:





# In[36]:


tablea=pd.read_csv('customer_details.csv', skipinitialspace = True)


# In[37]:


tableb=pd.read_csv('customer_policy_details.csv',skipinitialspace = True)


# In[38]:


tablea


# In[39]:


tableb


# In[ ]:





# In[60]:


Mastertable=pd.concat(table1,table2,axis=1)


# 

# In[47]:


Mastertable.groupby('Gender')['annual premium(in Rs)'].mean()


# In[15]:


Mastertable.groupby('Age')['annual premium(in Rs)'].mean()


# In[17]:


print(f"male to female ration is {round(table1['Gender'].value_counts()[0]/table1['Gender'].value_counts()[1],2)}")
print(f"generally, the standard is: \n balanced data ratio: {50/50}\n slightly balanced data ratio: {round(55/45,2)}-{60/40} \n imbalanced data ratio: {80/20}-{90/10}")


# In[ ]:


Mastertable.groupby('vehicle age')['annual premium(in Rs)'].mean()


# In[18]:


n = Mastertable['Age'].corr(Mastertable['annual premium(in Rs)'])
if n<-0.5:
    print("Strong negative relationship")
if n>0.5:
    print("Strong positive relationship")
if n>-0.5 and n<0.5:
    print("There is no relationship!")


# In[ ]:




