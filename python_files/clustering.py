import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Hierachical Clustering for verification
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

# Kmodes clustering
from kmodes.kmodes import KModes 

cluster_df = pd.read_csv('./csv/skill.csv')

# Kmodes clustering to identify the optimal K
def elbow_plot():
    cost = [] 
    K = range(1,11) 
    for k in list(K): 
        kmode = KModes(n_clusters = k, init = "random", n_init = 5, verbose = 1) 
        kmode.fit_predict(cluster_df) 
        cost.append(kmode.cost_) 
        
    plt.plot(K, cost, 'x-') 
    plt.xticks(K)
    plt.xlabel('No. of clusters') 
    plt.ylabel('Cost') 
    plt.title('Kmodes Clustering Elbow Plot') 
    plt.savefig('./images/ML/elbow.png')

# Hierachical clustering for verification
def categorical_hierachical():
    distance_matrix = pairwise_distances(cluster_df, metric='hamming')
    linkage_data = linkage(squareform(distance_matrix), method='complete')
    plt.figure(figsize = (8,6))
    dendrogram(linkage_data)
    plt.title('Hierarchical Clustering for Categorical Data')
    plt.gca().set_xticks([])
    plt.xlabel('Nodes')
    plt.ylabel('Hamming Distance')

    plt.axhline(0.28, linestyle='--', color='red')
    plt.axhline(0.245, linestyle='--', color='red')

    plt.text(530, 0.35, 'Suggested clusters: 4-6', weight='bold')
    plt.savefig('./images/ML/dendrogram.png')

elbow_plot() 
categorical_hierachical(); #clusters: 4-6

# Let's try to build a model with 6 clusters from the dendrogram and the elbow plot
def output_df():
    kmode = KModes(n_clusters=6, init = "random", n_init = 5, verbose=1)
    clusters = kmode.fit_predict(cluster_df)
    cluster_df['label'] = clusters
    cluster_df.to_csv('./csv/cluster_label.csv', index = False)

output_df()