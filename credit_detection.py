#!/usr/bin/env python
# coding: utf-8

# # CREDIT  CARD  FRAUD  DETECTION

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


value = pd.read_csv('creditcard.csv')


train_value, test_value = train_test_split(value, test_size=0.2, random_state=42)


train_inputs = train_value.drop('Class', axis=1)
train_outputs = train_value['Class']
test_inputs = test_value.drop('Class', axis=1)
test_outputs = test_value['Class']


scaler = StandardScaler()
train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.transform(test_inputs)


model = keras.Sequential([
    keras.layers.Dense(256, input_shape=(train_inputs.shape[1],), activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train_inputs, train_outputs, epochs=10, batch_size=32, validation_split=0.2)


test_loss, test_acc = model.evaluate(test_inputs, test_outputs)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
predictions = model.predict(test_inputs)
predictions = [1 if x > 0.5 else 0 for x in predictions]


print(classification_report(test_outputs, predictions))


# In[7]:


import os


# In[10]:


os.chdir("C:\\Users\\ronit\\Downloads")

