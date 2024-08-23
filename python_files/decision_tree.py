import pandas as pd
import matplotlib.pyplot as plt

cluster_df = pd.read_csv('./csv/cluster_label.csv')

X = cluster_df.drop('label', axis = 1)
y = cluster_df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.tree import DecisionTreeClassifier, plot_tree
dtree = DecisionTreeClassifier(max_depth= 3)
dtree = dtree.fit(X_train, y_train)

def draw_tree():
    plt.figure('Decision Tree', figsize= [20,8])
    plot_tree(dtree, fontsize= 10)
    plt.savefig('./images/ML/tree.png')

draw_tree()

# Internal testing & prediction
y_pred_tree = dtree.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred_tree) 
# accuracy score: 0.875

# Feature importance analysis
def importance():
    feature_imp = dtree.feature_importances_
    plt.bar([x for x in range(len(feature_imp))], feature_imp, edgecolor='black')
    plt.title('Feature Importance Summary')
    plt.xlabel("Feature")
    plt.ylabel("Score")
    plt.savefig('./images/ML/feature_importance.png')
    
importance()

# print column name from the feature importance analysis
colname = cluster_df.columns[[4, 7, 20,24]] # SQL, Python, Spark, Pipeline

# Save the model
import pickle
with open('./decision_tree_model.pkl', 'wb') as dtree_file:
    pickle.dump(dtree, dtree_file)