from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()

#개요


labels = pd.DataFrame(iris.target)
labels.columns=['labels']
data = pd.DataFrame(iris.data)
data.columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
data = pd.concat([data, labels], axis=1)

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(data, method='complete')

#Plot the dendrogram. using varieties as labels
plt.figure(figsize=(40, 20))
dendrogram(mergings,
           labels=labels.as_matrix(columns=['labels']),
           leaf_rotation=90,
           leaf_font_size=20,
           )
#plt.show()
from scipy.cluster.hierarchy import fcluster

predict = pd.DataFrame(fcluster(mergings,3,criterion='distance'))
print(predict)
predict.columns=['predict']

#print(test)
ct = pd.crosstab(predict['predict'],labels['labels'])
print(ct)
