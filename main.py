#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# # Description of the Dataset

# Our dataset is from Kaggle.com. It contains song data specifically features such as (artist name, popularity, key, release_date, etc). In total each song has 19 features given, and there are 107,654 songs (instances). Our target variable is a song recommendation based on past listening habits. Below, we load the dataset, try to locate null values, and plot some of the interesting data.

# In[4]:


# Load the dataset
data = pd.read_csv('../data.csv')


# In[5]:


print(data.head())


# In[6]:


# Check if there are any empty values in the dataset
data.isnull().values.any()


# In[7]:


data.isnull().sum()


# In[8]:


# Drop non-numeric columns
df_numeric = data.drop(columns=['artists', 'id', 'name', 'release_date'])

# Calculate correlation matrix
correlation_matrix = df_numeric.corr()

# Plot the correlation matrix using Seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title('Correlation Matrix of Sound Features')
plt.show()


# Using this heat map, we are able to see the correlation of different features on our dataset. This will help us in the future to use this features to hopefully recommend similar songs. From this we can see some interesting correlations, as with year and populatiry, or with loudness and energy that could be useful for the next steps in this project.

# In[9]:


# Plotting the distribution of the key attribute
plt.figure(figsize=(10, 6))
plt.hist(data['key'], bins=range(0, 13), alpha=0.7, edgecolor='black')
plt.title('Distribution of Musical Keys')
plt.xlabel('Key')
plt.ylabel('Frequency')
plt.xticks(range(0, 12))
plt.grid(axis='y', alpha=0.75)

# Display the plot
plt.show()


# We can see that even the lowest frequency of key 3 has about 7500 instances which is more than enough for testing purposes. This validates the quality of our dataset. It also gives us a sense of how balanced our data is, and from this we can infer that we are working with a good quality dataset.

# In[10]:


# Group by year and calculate the mean energy for each year
energy_trend = data.groupby('year')['energy'].mean()

# Convert index and values to numPy arrays
years = energy_trend.index.values
energy_means = energy_trend.values

# Plot the trend of energy over the years
plt.figure(figsize=(12, 6))
plt.plot(years, energy_means, linestyle='-')
plt.title('Trend of Energy in Songs Over the Years')
plt.xlabel('Year')
plt.ylabel('Mean Energy')
plt.grid(True)
plt.show()


# In[11]:


# Group by 'year' and calculate the mean loudness for each year
loudness_trend = data.groupby('year')['loudness'].mean()

# Convert index and values to numPy arrays
years = loudness_trend.index.values
loudness_means = loudness_trend.values

# Plot the trend of loudness over the years
plt.figure(figsize=(12, 6))
plt.plot(years, loudness_means, linestyle='-')
plt.title('Trend of Loudness in Songs Over the Years')
plt.xlabel('Year')
plt.ylabel('Mean Loudness')
plt.grid(True)
plt.show()


# Both of the graphs above trend energy and loudness, respectively, over the decades. This helps us understand how music has evolved and shifted throughout the decades, and helps us correlate these features with what a user would like. Since there are obivous trends in how the music in each decade sounds, we can maybe gain some insight into what decade of music to recommend to a user if we know specific patterns on the average loudness/energy or other feature of songs they listen to.
