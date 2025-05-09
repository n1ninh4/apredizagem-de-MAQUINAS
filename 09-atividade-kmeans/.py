import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

seeds_df = pd.read_csv('seeds.csv', header=None)

seeds_array = seeds_df.to_numpy()

X_train, X_test = train_test_split(seeds_array, test_size=0.3, random_state=42)

model = KMeans(n_clusters=3, random_state=42)

model.fit(X_train)

labels = model.predict(X_test)

x = X_test[:, 0]
y = X_test[:, 1]

plt.figure(facecolor='lightblue')
plt.scatter(x, y, c=labels, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clusters gerados pelo K-Means")
plt.show()