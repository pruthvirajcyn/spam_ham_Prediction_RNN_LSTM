#!/usr/bin/env python
# coding: utf-8

# # SPAM CLASSIFICATION with LSTM Network in Keras

# In[2]:


import pandas as pd 
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential


# ### Reading the data

# In[3]:


raw_data = pd.read_csv('./spam_train.csv', encoding='latin-1') 
raw_test_data = pd.read_csv('./spam_test.csv', encoding='latin-1')

print(raw_data.shape) 
print(raw_data.columns) 
print('\n')
print(raw_data.head(5)) 


# ### Check the labels and their frequencies

# In[4]:


classes = np.unique(raw_data['Label'], return_counts=True)
print(classes[0]) 
print(classes[1]) 


# ### Conver text to fixed length sequence

# In[12]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000) #Tokenizer is used to tokenize text
tokenizer.fit_on_texts(raw_data.Message) #Fit this to our corpus

x_train = tokenizer.texts_to_sequences(raw_data.Message) #'text to sequences converts the text to a list of indices
x_train = pad_sequences(x_train, maxlen=50) #pad_sequences makes every sequence a fixed size list by padding with 0s 
x_test = tokenizer.texts_to_sequences(raw_test_data.Message) 
x_test = pad_sequences(x_test, maxlen=50)

x_train.shape, x_test.shape # Check the dimensions of x_train and x_test  


# ### Prepare the target vectors for the network

# In[9]:


from tensorflow.keras.utils import to_categorical 
unique_labels = list(raw_data.Label.unique()) 
y_train = np.array([unique_labels.index(i) for i in raw_data.Label]) 
y_train = to_categorical(y_train) 
y_test = np.array([unique_labels.index(i) for i in raw_test_data.Label])
y_test = to_categorical(y_test)
y_test.shape


# In[11]:


import tensorflow.keras.backend as K 

def recall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    PP = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (PP + K.epsilon())
    return recall


# ### Building and training an LSTM model

# In[18]:


model=Sequential()
model.add(Embedding(10000,100))
model.add(LSTM(10,dropout=0.2))
model.add(Dense(3,activation='softmax'))


# In[19]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[20]:


model.fit(x_train,y_train,batch_size=32,
          epochs=1,validation_data=(x_test, y_test))


# In[ ]:




