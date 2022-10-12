import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

flags = pd.read_csv("flags.csv")
flags.head()
flags.describe()
flags.columnslabels = flags[['Landmass']]

labels = flags[['Landmass']]
data = flags[["Red","Green","Blue","Gold","White","Black","Orange"]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

model = DecisionTreeClassifier(random_state=1)
model.fit(train_data, train_labels)

score = model.score(test_data, test_labels)
print(score)
scores = []

for i in range(1,20):
    model = DecisionTreeClassifier(random_state=10,max_depth=i)
    model.fit(train_data, train_labels)
    score = model.score(test_data, test_labels)
    scores.append(score)

scores

plt.figure(figsize=(10,6))
plt.plot(range(1,20), scores)

plt.xlabel('max-depth')
plt.ylabel('score')
plt.title('Max depth vs Score: Color columns')

plt.xlim(0,20)
plt.xticks(np.arange(0, 20, step=2))
plt.show()


data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange","Circles",
              "Crosses","Saltires","Quarters","Sunstars","Crescent","Triangle"]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

scores = []
for i in range(1,20):
    model = DecisionTreeClassifier(random_state=1, max_depth=i)
    model.fit(train_data, train_labels)
    score = model.score(test_data, test_labels)
    scores.append(score)
    
scores

plt.figure(figsize=(10,6))
plt.plot(range(1,20), scores)

plt.xlabel('max-depth')
plt.ylabel('score')
plt.title('Max depth vs Score: Color and Shape columns')

plt.xlim(0,20)
plt.xticks(np.arange(0, 20, step=2))
plt.show()