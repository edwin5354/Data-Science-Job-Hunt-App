import pandas as pd

# NLP library
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# Open csv
df = pd.read_csv('./csv/raw.csv')

# Create a skill list 
skill_list = ['R', 'RStudio', 'Markdown', 'Latex', 'SparkR', 'D3', 'D3.js','Microsoft Office','Excel',
    'Unix', 'Linux', 'MySQL', 'Microsoft SQL server', 'SQL','VBA','Qlik',
    'Python', 'SPSS', 'SAS', 'C#','Matlab','Java', 'keras',
    'JavaScript', 'HTML', 'HTML5', 'CSS', 'CSS3','PHP', 'Tableau',
    'AWS', 'Amazon Web Services ','Google Cloud Platform', 'GCP','theano',
    'Microsoft Azure', 'Azure', 'Hadoop', 'Spark', 'JIRA', 'Node.js',
    'MapReduce', 'Map Reduce','Shark', 'Cassandra', 'ETL', 'pipeline',
    'NoSQL', 'MongoDB', 'GIS', 'Haskell', 'Scala', 'Ruby','Perl', 'spark', '.NET',
    'Mahout', 'Stata','Deep Learning','Machine Learning',  'Pytorch', "Tensorflow",'Caffe','API','seo','Business Intelligence',
    'BI']

df_size = len(df)

skill_df = pd.DataFrame(0, index = df.index, columns = [skill.lower() for skill in skill_list])

for i in range(df_size):
    test_text = df['description'][i]
    word_token = word_tokenize(test_text)

    customStopWords = set(stopwords.words('english') + list(punctuation))
    filtered_text = [word.lower() for word in word_token if word.lower() not in customStopWords]

    for each_skill in skill_df.columns:
        if each_skill in filtered_text:
            skill_df.at[i, each_skill] = 1

# Drop some uneccesary columns
skill_df = skill_df.loc[:, (skill_df.sum() != 0)]
skill_df.to_csv('./skill.csv', index=False)

merged_df = df.join(skill_df)
merged_df = merged_df.drop(['description'], axis= 1)
merged_df.to_csv('./csv/merged.csv', index = False)