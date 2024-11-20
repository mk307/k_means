import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading Dataset
data1 = pd.read_csv("toy_dataset.csv")
data2 = data1.loc[:1000, ['Age','Income']]

# Finding minimum and maximum value for each feature
minmax = MinMaxScaler()
minmax.fit(data2)
data3 = minmax.transform(data2)
data4 = pd.DataFrame(data3)
data4.columns =['Age', 'Income'] 

# Scatterplot between continuous features: Age and Income
plt.scatter(data4['Age'], data4['Income'], c ='black')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Elbow Method to determine optimal k value 
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data4)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Choosing k value 
K = 3

# Randomly picking observations as centroids
centroids = (data4.sample(n = K))
plt.scatter(data4['Age'], data4['Income'], c ='black')
plt.scatter(centroids['Age'], centroids['Income'], c ='red')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

diff = 1
j = 0

while(diff != 0):
    XD = data4
    i = 1
    for index1,row_c in centroids.iterrows():
        ED = []
        for index2,row_d in XD.iterrows():
            d1 = (row_c['Age'] - row_d['Age'])**2
            d2 = (row_c['Income'] - row_d['Income'])**2        
            d = np.sqrt(d1 + d2)
            ED.append(d)
        data4[i] = ED
        i = i+1

    C = []
    for index,row in data4.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos = i+1
        C.append(pos)
    
    data4["Cluster"] = C
    centroids_new = data4.groupby(["Cluster"]).mean()[["Age","Income"]]
    if j == 0:
        diff = 1
        j = j+1
    else:
        diff = (centroids_new['Age'] - centroids['Age']).sum() + (centroids_new['Income'] - centroids['Income']).sum()
        print(diff.sum())
    centroids = data4.groupby(["Cluster"]).mean()[["Age","Income"]]
 
# Updated centroids
color = ['cyan', 'purple', 'yellow']
for k in range(K):
    data5 = data4[data4["Cluster"] == k+1]
    plt.scatter(data5['Age'], data5['Income'], c = color[k])
centroids = (data5.sample(n = K))
plt.scatter(centroids_new['Age'], centroids_new['Income'], c ='red')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Deploying Logistic Regression to predict cluster assigned value 
data6 = data4.loc[:1000, ['Age','Income', 'Cluster']]
X = data6.drop(['Cluster'], axis=1)
y = data6['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 

model = LogisticRegression()
model.fit(X_train, y_train)

acc_score_train = accuracy_score(y_train, model.predict(X_train))
print('Training Accuracy = ', acc_score_train)
acc_score_test = accuracy_score(y_test, model.predict(X_test))
print('Testing Accuracy = ', acc_score_test)



