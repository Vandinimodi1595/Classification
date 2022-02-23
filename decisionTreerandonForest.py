#Decision Tree Classifier vs Random Forest Classifier

import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import pydotplus
import graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from dtreeviz.trees import dtreeviz  # remember to load the package
from sklearn.datasets import load_iris
from  io import StringIO
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")

#conda install -c conda-forge pydotplus -y
#conda install -c conda-forge python-graphviz -y


if __name__=='__main__':
    df = pd.read_csv('/Users/vandinimodi/Desktop/Sem3/CPSC8650DataMining/Git/DecisionTreeClassifier/drug200.csv')
    #various way of data exploration
    print(df.info())
    print(df.shape)
    print(df[0:10])
    #count of each categorical value
    print("-----------------------")
    for i in df.columns:
        if i not in ('Age','Cholesterol','Na_to_K'):
            print(df[i].value_counts())
    #as numpy array
    x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    print(len(x))
    print(type(x))
    print(df.values)
    print(df.head())
    print(df.describe().T)
    columns = df.columns
    print(columns)
    grouped_df = df.groupby(['Drug']).agg('count')
    plt.bar(grouped_df.index,height = grouped_df['Age'])
    plt.xlabel('Drug')
    plt.ylabel('Count')
    #plt.show()
    print(df['Sex'].unique())
    print(df['BP'].unique())

    #data preprocessing
    #using Encoder
    sex_enc = LabelEncoder()
    sex_enc.fit(['F','M']) #sex_enc.fit(df['Sex'].unique())
    df['Sex'] = sex_enc.transform(df['Sex'])

    bp_enc = LabelEncoder()
    bp_enc.fit(['LOW', 'NORMAL', 'HIGH'])
    df['BP'] = bp_enc.transform(df['BP'])

    cho_enc = LabelEncoder()
    cho_enc.fit(['LOW', 'NORMAL', 'HIGH'])
    df['Cholesterol'] = cho_enc.transform(df['Cholesterol'])

    drug_enc = LabelEncoder()
    drug_enc.fit(['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
    df['Drug'] = drug_enc.transform(df['Drug'])

    print(df.head())

    #using get_dummies
    df1 = df.copy()
    dummy = pd.get_dummies(data=df1[['Sex', 'BP', 'Cholesterol']], drop_first=True)
    df1 = df1.drop(['Sex', 'BP', 'Cholesterol'], axis=1)
    df1 = pd.concat([df1, dummy], axis=1)
    print(df1.head())

    #data visualization from given data
    sns.pairplot(data=df1, hue='Drug', palette='pink')
    #plt.show()

    plt.figure(figsize=(8, 8))
    sns.heatmap(df1.corr(), annot=True, fmt='.3f', cmap='BuPu_r')
    plt.title("CORRELATION BETWEEN FEATURES", fontsize=18, color='black', pad=20)
    #plt.show()

    #setup decision tree
    x_train, x_test, y_train, y_test = train_test_split(df.drop(['Drug'], axis=1), df['Drug'], train_size=0.7,random_state=11)
    print("Shape of the training set X is", x_train.shape)
    print("Shape of the training set Y is", y_train.shape)
    print(np.unique(y_train))

    #modelling
    dt = DecisionTreeClassifier(criterion='gini',max_depth=3)
    dt.fit(x_train,y_train)

    #prediction
    pt = dt.predict(x_test)
    print(pt[0:10])
    print(y_test.head(5))

    #accuracy check
    print("Accuracy of the testing decision tree is:", accuracy_score(y_test, pt))

    #tuning best hyperparameters using grid search
    best_param = {'criterion': ['gini','entropy'], 'max_depth' : [2,3,4]}
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=best_param,cv=5)
    grid_search.fit(x_train,y_train)
    print("-------------------------------------------")
    print("grid search parameters and estimators are:")
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    print("-------------------------------------------")

    #testing with best parameters
    tuned_pt = dt.predict(x_test)
    print("Accuracy after tuning with grid search:",accuracy_score(y_test,tuned_pt))

    #randomForestClassifier
    rfc = RandomForestClassifier(max_depth=3,random_state=11)
    rfc.fit(x_train,y_train)

    #prediction RFC
    rfc_pt = rfc.predict(x_test)
    print("Accuracy for Random Forest Classifier is:",accuracy_score(y_test,rfc_pt))

    #visulizing decision tree structure
    #text representation
    text_representation = tree.export_text(dt)
    #print(text_representation)

    #graphviz
    dot_data = StringIO()
    filename = "dt.png"
    featureNames = df.columns[0:5]
    targetnames = df.drop(['Age','Sex', 'BP', 'Cholesterol','Na_to_K'], axis=1)
    df['Drug'].unique()
    dot_data = StringIO()
    filename = "drugtree.png"
    featureNames = df.columns[0:5]
    df['Drug'].unique()

    np.unique(y_train)
    iris = load_iris()
    #tree.export_graphviz(dt)
    df.columns.values[:5]  #['Age' 'Sex' 'BP' 'Cholesterol' 'Na_to_K']
    dot = tree.export_graphviz(dt, out_file=None,
                               feature_names=df.columns.values[:5],
                               class_names=["4", "3","2","0","1"],
                               filled=True, rounded=True,
                               special_characters=True)

    # we create a graph from dot source using graphviz.Source
    graph = graphviz.Source(dot)
    #print(graph)

    dot_data = StringIO()
    filename = "drugtree.png"
    featureNames = df.columns[0:5]
    out = tree.export_graphviz(dt, feature_names=df.columns.values[:5], out_file=dot_data,
                               class_names=["4", "3","2","0","1"], filled=True, special_characters=True, rotate=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(100, 200))
    plt.imshow(img, interpolation='nearest')





