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
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist


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


# In[18]:


fig, axs = plt.subplots(nrows=3, ncols=4, constrained_layout=True, figsize=(20,15))
sns.histplot(ax=axs[0][0],x=data["popularity"],color="grey")
sns.histplot(ax=axs[0][1],x=data["duration_ms"],color="red")
sns.histplot(ax=axs[0][2],x=data["danceability"],color="blue")
sns.histplot(ax=axs[0][3],x=data["energy"],color="purple")
sns.histplot(ax=axs[1][0],x=data["loudness"],color="black")
sns.histplot(ax=axs[1][1],x=data["speechiness"],color="red")
sns.histplot(ax=axs[1][2],x=data["acousticness"],color="orange")
sns.histplot(ax=axs[1][3],x=data["instrumentalness"],color="yellow")
sns.histplot(ax=axs[2][0],x=data["liveness"],color="green")
sns.histplot(ax=axs[2][1],x=data["valence"],color="brown")
sns.histplot(ax=axs[2][2],x=data["tempo"],color="magenta")
sns.histplot(ax=axs[2][3],x=data["year"],color="indigo")
plt.show()


# Both of the graphs above trend energy and loudness, respectively, over the decades. This helps us understand how music has evolved and shifted throughout the decades, and helps us correlate these features with what a user would like. Since there are obivous trends in how the music in each decade sounds, we can maybe gain some insight into what decade of music to recommend to a user if we know specific patterns on the average loudness/energy or other feature of songs they listen to.

# In[13]:


# Select only the numerical features for clustering
# Includes year, but could also remove to prevent listening to certain eras 
features_to_cluster = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
]

# Extract the relevant numerical features
X = data[features_to_cluster]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[14]:


# Define a unique seed for reproducibility
seed = 42
kmeans = KMeans(n_clusters=15, random_state=seed)

# Fit the K-means model
kmeans.fit(X_scaled)

# Assign the cluster labels to each song
data['cluster_label'] = kmeans.labels_


# In[15]:


pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()


# In[16]:


# Get centroids of each cluster
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=features_to_cluster)

# Analyze the centroids to understand each cluster
print(centroids_df)
