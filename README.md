# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.k cluster centroids randomly from the dataset.

2.Assign each customer to the nearest centroid based on a distance measure (e.g., Euclidean distance).

3.Update each centroid to the mean position of all customers assigned to it.

4.Repeat steps 2 and 3 until centroids no longer change significantly, then use the resulting clusters to segment the customers. 
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:Kavipriya S.P 
RegisterNumber: 2305002011 
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

#Extract features
X=data[['Annual Income (k$)','Spending Score (1-100)']]

plt.figure(figsize=(4,4))
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(X)
centroids=Kmeans.cluster_centers_


labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

colors=['r','g','b']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![Screenshot 2024-10-25 220725](https://github.com/user-attachments/assets/77006037-3933-4b27-a8ba-38cb4dad05f2)
![Screenshot 2024-10-25 220737](https://github.com/user-attachments/assets/13fdf1df-1019-40b0-8875-e07db09628f6)
![Screenshot 2024-10-25 220745](https://github.com/user-attachments/assets/20711f83-0df2-4a92-bc4d-7d3142044f7c)
![Screenshot 2024-10-25 221126](https://github.com/user-attachments/assets/ab67899d-6717-4d8c-a8a1-dc60ee22e36e)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
