import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

p = pd.read_csv('C:\\users\\thecasual\\Documents\\Github\\projects\\web.csv')
p = p.drop(columns='timestamp')
p = p[['url', 'hour_of_day']]
#df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# Create a bunch of new columns because we need numbers!
#X = pd.get_dummies(p, columns=p.columns, drop_first=True).to_numpy()

# https://towardsdatascience.com/outlier-detection-with-extended-isolation-forest-1e248a3fe97b
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html
#print(type(j))
#j.to_csv('C:\\users\\thecasual\\Documents\\Github\\projects\\test.csv')

# define % of anomalies


X = pd.get_dummies(p, columns=p.columns, drop_first=True).to_numpy()
X_train, X_test = train_test_split(df, test_size=0.2)
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')

clf.fit(X_train)