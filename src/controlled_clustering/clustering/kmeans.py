import numpy as np

class Kmeans:
    def __init__(self, k, lb, ub):
        lb=np.asarray(lb,dtype=float).reshape(-1,)
        ub=np.asarray(ub,dtype=float).reshape(-1,)
        self.lower_bound=lb;self.upper_bound=ub;
        self.k=k
        n=lb.shape[0]
        self.samples=[]
        self.dim=n        
        self.centroids=np.zeros((k,n),dtype=float)
        for i in range(k):
            self.centroids[i,:]=np.random.rand(n)*(ub-lb)+lb

        self.probabilities=np.zeros(k,dtype=float)
        self.counts=np.zeros(k,dtype=int)

    def accuracy(self):
        d=0
        for s in self.samples:
            s=np.asarray(s,dtype=float).reshape(self.dim,)
            distances=np.zeros(self.k,dtype=float)
            for i in range(self.k):            
                distances[i]=np.linalg.norm(self.centroids[i,:]-s)                  
            closest_centroid=np.argmin(distances)
            d+=distances[closest_centroid]
        return d/len(self.samples) if self.samples else 0.0
    
    def step(self,x):
        self.samples.append(x)
        x=np.asarray(x,dtype=float).reshape(self.dim,)
        distances=np.zeros(self.k,dtype=float)
        for i in range(self.k):            
            distances[i]=np.linalg.norm(self.centroids[i,:]-x)                  
        closest_centroid=np.argmin(distances)
        self.counts[closest_centroid]+=1
        self.probabilities=self.counts/np.sum(self.counts)
        self.centroids[closest_centroid,:]=(self.centroids[closest_centroid,:]*(self.counts[closest_centroid]-1)+x)/self.counts[closest_centroid]
    
    def __repr__(self):
        return "\n".join(
            f"Cluster {i+1}:\n"
            f"Centroid={self.centroids[i,:]}\n"
            f"Probability={self.probabilities[i]:.4f}\n"
            f"Count={self.counts[i]:.4f}"
            for i in range(self.k)
        )