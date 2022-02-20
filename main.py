import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



if __name__=='__main__':
    df = pd.read_csv('/Users/vandinimodi/Desktop/Sem3/CPSC8650DataMining/Git/DecisionTreeClassifier/drug200.csv')
    print(df.info())
    print(df.head())
    print(df.describe().T)
    columns = df.columns
    print(columns)
    grouped_df = df.groupby(['Drug']).agg('count')
    plt.bar(grouped_df.index, height=grouped_df['Age'])
