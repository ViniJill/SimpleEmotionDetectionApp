#!/usr/bin/env python
# coding: utf-8

# # Simple Emotion Classifier App
# 
# ### - Emotion Detection based on Text
# ### - Text Classifier

# In[3]:


# Load EDA Packages
import pandas as pd
import numpy as np

# Import Data Viz Packages
import seaborn as sns

# Load Text Cleaning Packages
import neattext.functions as nfx

# Load ML Packages
# -Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# -Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[11]:


# Load datasets
df = pd.read_csv("emotion_dataset.csv")


# In[12]:


df.head()


# In[13]:


# Value Counts
df['Emotion'].value_counts()


# In[14]:


sns.countplot(x='Emotion', data = df)


# In[18]:


# Data Cleaning

# Remove User Handles
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)


# In[19]:


# Remove Stopwords
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)


# In[22]:


# Remove Special Characters
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_special_characters)


# In[23]:


df


# In[24]:


# Features & Labels
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']


# In[25]:


# Split data
x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size = 0.3, random_state = 42)


# In[26]:


# Build Pipeline
from sklearn.pipeline import Pipeline


# In[32]:


# Logistic Regression Pipeline
pipe_lr = Pipeline(steps = [('cv', CountVectorizer()),('lr', LogisticRegression())])


# In[33]:


# Train and Fit Data
pipe_lr.fit(x_train, y_train)


# In[35]:


pipe_lr


# In[34]:


# Check Accuracy
pipe_lr.score(x_test, y_test)


# In[37]:


# Make Prediction
ex1 = "This book was so interesting and it made me awesome"


# In[38]:


pipe_lr.predict([ex1])


# In[39]:


# Prediction probability
pipe_lr.predict_proba([ex1])


# In[40]:


# To know the classes
pipe_lr.classes_


# In[41]:


# Save model & Pipeline
import joblib
pipeline_file = open("emotion_classifier_pipe_lr_3_sept_2021.pkl","wb")
joblib.dump(pipe_lr, pipeline_file)
pipeline_file.close()

