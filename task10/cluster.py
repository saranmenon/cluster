import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
import pylab as pl

var=pd.read_csv("estate.csv")
x=var[["zip"]]
y=var[["price"]]

kmeans=KMeans(n_clusters=7)
kmeansoutput=kmeans.fit(y)

pl.figure("n_cluster k-means")
pl.scatter(x,y,c=kmeansoutput.labels_)
pl.xlabel("x axis")
pl.ylabel("y axis")
pl.title("cluster")
pl.show()
