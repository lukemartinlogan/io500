
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import os,sys

class Dataset:
    def __init__(self, df=None, clusters=None):
        self.df = df
        self.clusters = clusters
        self.centers = None

    def read_csv(self, path):
        self.df = pd.read_csv(path)
        return self

    def to_csv(self, path):
        self.df.to_csv(path, index=False)

    def load_features(self, path) -> list:
        return list(pd.read_csv(path, header=None).iloc[:,0])

    def _basic_stats(self, df, n) -> dict:
        return {
            "n": len(df),
            "rel_n": len(df)/n,
            "mean": df.mean(),
            "std": df.std(),
            "0%": df.min(),
            "25%": df.quantile(q=.25),
            "50%": df.median(),
            "75%": df.quantile(q=.75),
            "90%": df.quantile(q=.9),
            "100%": df.max()
        }

    def divide(self, cluster_col="cluster"):
        df = self.df
        df.sort_values(by=cluster_col,inplace=True)
        cluster_ids = df[cluster_col].unique().tolist()
        clusters = { cluster_id:df[df[cluster_col] == cluster_id] for cluster_id in cluster_ids }
        return Dataset(df=df, clusters=clusters)

    def analyze(self, features) -> dict:
        if type(features) == type(""):
            features = [features]
        if self.clusters == None:
            return { feature: self._basic_stats(self.df[feature], len(self.df)) for feature in features }
        else:
            return { feature: {cluster_id: self._basic_stats(cluster_df[feature], len(self.df)) for cluster_id,cluster_df in self.clusters.items()} for feature in features }

    def print(self) -> None:
        if self.clusters == None:
            print(self.df)
        else:
            clusters = self.clusters
            for cluster_id,df in clusters.items():
                print("CLUSTER: {}".format(cluster_id))
                print(df)
                print()

    def groupby(self, features, cluster_col="cluster"):
        self.df.loc[:,cluster_col] =  self.df.groupby(features).ngroup()
        return self

    def partition(self, feature, step=.1, scale=10, min_range=100, cluster_col="cluster"):
        df = self.df
        feature_df = df[feature]
        q = step
        min_value = feature_df.min()
        df[cluster_col] = 0
        id = 0
        while True:
            max_value = feature_df.quantile(q)*scale
            if (max_value - min_value) < min_range:
                max_value = min_value + min_range
            df.loc[(min_value <= df[feature]) & (df[feature] < max_value),cluster_col] = id
            if q == 1:
                break
            min_value = max_value
            q = stats.percentileofscore(feature_df,max_value)/100+step
            if q > 1:
                q = 1
            id += 1

    def kmeans(self, features, k=10, cluster_col="cluster", return_centers=False):
        if type(features) == type(""):
            features = [features]

        #Standardize Dataframe Features
        feature_df = RobustScaler().fit_transform(self.df[features])

        #Create clusters
        if len(feature_df) < k:
            k = len(feature_df)
        km = KMeans(n_clusters=k, verbose=10)
        clusters = np.array(km.fit_predict(feature_df))
        centers = km.cluster_centers_

        #Set clusters in the dataframe
        self.df.loc[:,cluster_col] = clusters
        if return_centers:
            return centers
        else:
            return self

    def agglomerative(self, features, max_k = 200, dist_thresh=None, cluster_col="cluster"):
        if type(features) == type(""):
            features = [features]

        #A simple distance threshold estimate
        if dist_thresh==None:
            dist_thresh = np.sqrt(len(features)/4)

        #Run KMeans with high k and extract cluster centers
        centers = self.kmeans(features, max_k, cluster_col="$tempcol", return_centers=True)
        centers = pd.DataFrame(data=centers)

        #Run agglomerative clustering on the cluster centers
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh).fit(centers)
        labels = agg.labels_
        k = agg.n_clusters_

        #Re-label each cluster
        self.df.loc[:,cluster_col] = 0
        for cluster, label in zip(range(max_k), labels):
            self.df.loc[df["$tempcol"] == cluster, cluster_col] = label
        self.df = self.df.drop(columns="$tempcol")
        return self

    def random_sample(self, n) -> pd.DataFrame:
        df = self.df
        n = int(n)
        if len(df) >= n:
            sample = df.sample(n, replace=False)
            return (sample, df.drop(sample.index))
        else:
            return (df.sample(n, replace=True), df)

    def stratified_random_sample(self, weights:list) -> tuple:
        clusters = self.clusters
        dfs = list(zip(*[random_sample(df,len(df)*weight) for df,weight in zip(clusters.values(),weights)]))
        return (pd.concat(dfs[0]), pd.concat(dfs[1]))
