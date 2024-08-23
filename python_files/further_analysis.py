import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from dython.nominal import associations

# Open csv
skill_df = pd.read_csv('./csv/skill.csv')

def skill_chart():
    sum_df = pd.DataFrame(data = skill_df.sum(axis=0), columns= ['count'])
    sum_df.index.name = 'skill'

    # find skills with top 5 occurences
    sns.barplot(data = sum_df.nlargest(5, 'count'), x = 'count', y = 'skill', edgecolor='black', color='r', orient='h')
    plt.xlabel('Frequency')
    plt.ylabel('Hard Skill')
    plt.title('Top 5 essential technical skills for the data science industry')
    plt.savefig('./images/data_visualisation/skills.png')

skill_chart()

# Correlation Matrix
def corr_matrix():
    associations(skill_df, plot=True, annot=False, figsize=(12, 10), cmap='viridis')
    plt.title('Association Matrix for Skills', fontsize=16)
    plt.xticks(rotation=0, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.savefig('./images/data_visualisation/association_matrix.png')

corr_matrix()

# Frequency of clusters
cluster_df = pd.read_csv('./csv/cluster_label.csv')
def show_label():
    cluster_count = pd.DataFrame(data = cluster_df['label'].value_counts(), columns= ['count'])
    cluster_count.index.name = 'Cluster Type'
    sns.barplot(data = cluster_count, x = 'Cluster Type', y = 'count', color='green', edgecolor='black')
    plt.xlabel('Cluster Type')
    plt.ylabel('Count')
    plt.title('Cluster distribution from KModes')
    plt.savefig('./images/ML/cluster.png')

show_label()

# Save a new csv that stores the job info and the clusters only
def new_df():
    cluster_label_df = pd.read_csv('./csv/cluster_label.csv')
    label_df = cluster_label_df['label']

    raw_data = pd.read_csv('./csv/raw.csv')
    job_df = raw_data.drop(['work-type', 'description'], axis = 1)

    cluster_merged = job_df.join(label_df)
    cluster_merged.to_csv('./csv/cluster_merged.csv')

new_df()