import streamlit as st
import pandas as pd
import pickle

path = "./csv/skill.csv"
skill_df = pd.read_csv(path)

music = "Rachmaninoff Elegie Op. 3 No. 1.mp3"
st.write('Rachmaninoff Elegie Op. 3 No. 1')
st.audio(music, format="audio/mp3", loop=True)

st.image('./images/bg_img.png')

st.title('Job Skill Hunter App')

st.write("The data was gathered exclusively from jobsDB. Natural Language Processing (NLP) was employed to sift through the text and pinpoint shared skills in the realm of data science. The application leverages unsupervised machine learning techniques such as K-mode clustering to spotlight the key technical skills currently sought after in the industry.")

st.subheader('Exploratory Data Analysis')

st.image("./images/data_visualisation/skills.png")
st.markdown(
"""    
After delving into the key job skills in the data science domain, here are the top 5 sought-after hard skills:

- Python programming is a prerequisite in nearly 50 percent of the data science job listings on jobsDB.
- SQL proficiency stands out as an essential skill for managing and analyzing data within relational databases.
- Proficiency in business intelligence tools, R programming, and Amazon Web Services are also in high demand within the data science realm.
"""
)

st.image('./images/data_visualisation/association_matrix.png')
st.write(
"The Dython library is used to create a correlation matrix for categorical variables. In the matrix, shades of green and yellow indicate stronger positive correlations between skills, while shades of blue suggest potential negative correlations. This color gradient helps to quickly visualize the relationships between different skills."
)

st.subheader('K-mode Clustering')
st.write('Because the skills are categorical in nature, K-mode clustering and hierarchical clustering were applied to delineate the clusters within the dataset based on these categorical values.')

st.image("./images/ML/elbow.png")
st.write("By applying K-mode clustering, the optimal number of clusters for this dataset was identified. Although the elbow plot is a heuristic method, it indicates that the optimal number of clusters lies between 4 and 6, as evidenced by the noticeable decrease in cost within this range.")

st.image('./images/ML/dendrogram.png')
st.write("In contrast to typical hierarchical clustering, which typically employs Euclidean distance to determine proximity between data points, in this particular case where the variables are categorical, Hamming distance is the chosen metric for analysis. The dendrogram is then generated, suggesting an optimal cluster range of 4-6 based on the analysis.")

st.image('./images/ML/cluster.png')
st.write("The K-Mode clustering algorithm was reapplied to classify each datapoint into 6 clusters. The resulting bar plot illustrates the distribution of the datapoint types across the clusters. This distribution will be leveraged for supervised machine learning to analyze feature importance within each cluster, pinpointing the key features that contribute to the distinctions among them.")

st.subheader('Decision Tree Classification Model')

st.image("./images/ML/tree.png")
st.write("A decision tree classifier was developed to pinpoint and forecast the job cluster for each instance. The dataset is divided, with 75 percent allocated for training and 25 percent for testing. The accompanying figure showcases a decision tree constructed using the Gini method, illustrating cause-and-effect relationships in a straightforward manner, simplifying complex processes.")
st.write("After completing the validation process, the model demonstrated an accuracy score of 87.5%. After the decision tree model was developed, it was saved for future predictive analysis to recommend job search opportunities.")

st.image("./images/ML/feature_importance.png")
st.markdown(
"""
Additionally, a feature importance analysis was performed, revealing that 4 out of the 34 features play a crucial role for prediction. These features include:
- Feature 4: SQL
- Feature 7: Python
- Feature 20: Spark
- Feature 24: Data Pipeline            
"""
)

st.subheader('Job Search Engine')
st.write("You can find out what jobs would be suitable for your current data science skills by ticking the checkboxes below.")

skill_list = skill_df.columns.tolist()
check_list = []

for i, skill in enumerate(skill_list):
    if i % 5 == 0:
        cols = st.columns(5)
    checked = cols[i % 5].checkbox(skill)
    check_list.append(1 if checked else 0)

def input_features(skill_select):
    data = dict(zip(skill_list, skill_select))
    features = pd.DataFrame(data, index=[0])
    return features

# Open the saved model
pickle_path = "decision_tree_model.pkl"

with open(pickle_path, 'rb') as file:
    saved_model = pickle.load(file)

user_df = input_features(check_list)

# Open the clustere_merged csv to show the job details relating to the clusters
merged_label_df = pd.read_csv("./csv/cluster_merged.csv", index_col= 0)

if st.button("Search"):
    prediction = saved_model.predict(user_df)
    st.write(f'Predicted Output: Cluster {prediction[0]}') 
    show_df = merged_label_df.loc[merged_label_df['label'] == prediction[0]]
    show_df = show_df.drop(['label'], axis = 1)
    st.dataframe(show_df)
