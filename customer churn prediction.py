#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv(r"C:\Users\admin\OneDrive\Documents\archive (18)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df


# In[3]:


df.info()


# In[4]:


df.drop("customerID" , axis = 1 , inplace = True)


# In[5]:


df.dtypes


# In[6]:


df.TotalCharges.values


# In[7]:


df[df["TotalCharges"] == ' ']


# In[8]:


df = df[df["TotalCharges"] != ' ']
df


# In[9]:


df["TotalCharges"] = df["TotalCharges"].astype(float)


# In[10]:


df.dtypes


# In[11]:


for column in df:
    if df[column].dtypes == "object":
        print(f" {column} : {df[column].unique()}")


# In[12]:


df.replace("No internet service" , "No" , inplace = True)


# In[13]:


df.replace("No phone service" , "No" , inplace = True)


# In[14]:


yes_no_columns = ["Partner" , "Dependents" , "PhoneService" , "MultipleLines" , "OnlineSecurity" , "OnlineBackup", 
                  "DeviceProtection" , "TechSupport" , "StreamingTV" , "StreamingMovies","PaperlessBilling","Churn"]

for i in yes_no_columns:
    df[i].replace({"Yes":1 , "No":0} , inplace = True)


# In[15]:


df["gender"].replace({"Male":1 , "Female":0} , inplace = True)


# In[16]:


for column in df:
    if df[column].dtypes == "object":
        print(f" {column} : {df[column].unique()}")


# In[17]:


df = pd.get_dummies(df).astype(int)


# In[18]:


df


# In[19]:


from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
df["tenure"]= ms.fit_transform(df[["tenure"]])
df["MonthlyCharges"]= ms.fit_transform(df[["MonthlyCharges"]])
df["TotalCharges"]= ms.fit_transform(df[["TotalCharges"]])


# In[20]:


x = df.drop("Churn" , axis = 1)
y = df["Churn"]


# In[21]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 ,random_state = 42)


# In[22]:


print(x_train.shape)
print(x_test.shape)


# In[23]:


model = keras.Sequential([
    keras.layers.Dense(10 ,input_shape = (26,) ,activation = 'relu'),
    keras.layers.Dense(1 , activation = "sigmoid")
])

model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ['accuracy']
)

model.fit(x_train , y_train , epochs = 100)


# In[24]:


model.evaluate(x_test , y_test)


# In[25]:


y_pred = model.predict(x_test)


# In[26]:


y_pred[:10]


# In[27]:


yp = []
for i in y_pred:
    if i > 0.1:
        yp.append(1)
        
    else:
        yp.append(0)


# In[28]:


yp[:10]


# In[29]:


y_train[:10]


# # precision and recall

# #### p = Tp/Tp+Tn

# #### R = Tp/Tp+Fn

# # conventional neural network

# In[31]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# In[32]:


(x_train  , y_train ), (x_test , y_test) = keras.datasets.cifar10.load_data()


# In[33]:


x_train.shape , x_test.shape


# In[34]:


plt.figure(figsize = (12,1))
plt.imshow(x_train[0])


# In[35]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[36]:


y_train = y_train.reshape(-1)
y_train[:5]


# In[37]:


def show_image(x,y,index):
    plt.figure(figsize = (12,1))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])
    plt.show


# In[38]:


show_image(x_train , y_train , 0)


# In[39]:


x_train = x_train/255
x_test = x_test/255


# In[40]:


ann = keras.Sequential([
    keras.layers.Flatten(input_shape = (32,32,3)),
    keras.layers.Dense(3000 , activation = 'relu'),
    keras.layers.Dense(1000 , activation = 'relu'),
    keras.layers.Dense(10 , activation = 'sigmoid')
])

ann.compile(
    optimizer = 'adam',
    loss = "sparse_categorical_crossentropy",
    metrics = ['accuracy']
)

ann.fit(x_train , y_train , epochs = 5)


# In[41]:


ann.evaluate(x_test , y_test)


# In[42]:


from sklearn.metrics import classification_report

y_pred = ann.predict(x_test)
y_pred_class = [np.argmax(i) for i in y_pred]

print(classification_report(y_test , y_pred_class))


# In[44]:


cnn = keras.Sequential([
   #cnn
    keras.layers.Conv2D(filters = 32 , kernel_size = (3,3) , input_shape = (32,32,3) , activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters = 32 , kernel_size = (3,3) , input_shape = (32,32,3) , activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    
    # dense
    keras.layers.Flatten(),
    keras.layers.Dense(60 , activation = 'relu'),
    keras.layers.Dense(10 , activation = 'sigmoid')
])


# In[45]:


cnn.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

cnn.fit(x_train , y_train , epochs = 10)


# In[46]:


cnn.evaluate(x_test , y_test)


# In[48]:


from sklearn.metrics import classification_report

y_pred = cnn.predict(x_test)
y_pred_class = [np.argmax(i) for i in y_pred]

print(classification_report(y_test , y_pred_class))


# In[ ]:




