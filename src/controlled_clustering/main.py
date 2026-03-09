import numpy as np
import matplotlib.pyplot as plt
from controlled_clustering.data.datastream import DataStream as ds
from controlled_clustering.clustering.kmeans import Kmeans as km
from controlled_clustering.controller.mpc import MPC as mpc

if __name__=="__main__":
    path="D:\\controlled-clustering\\src\\controlled_clustering\\data\\data.csv"
    ns=24;k=4;my=5;mu=4;N=4;T=5;dev=1e-1;outer_steps=100
    lb=np.zeros((ns,1),dtype=float)
    ub0=10*np.ones((ns,1),dtype=float)
    umin=-dev*np.ones((ns,1),dtype=float)
    umax=+dev*np.ones((ns,1),dtype=float)

    data_stream_open=ds(path)
    data_stream_closed=ds(path)
    kmeans_open=km(k,lb,ub0)
    kmeans_closed=km(k,lb,ub0)
    controller=mpc(ny=k,my=my,nu=ns,mu=mu,N=N,umin=umin,umax=umax)

    dso=[];dsc=[]
    eso=[];esc=[]
    outer=0
    while outer<outer_steps and data_stream_open.has_next_sample():
        sample=data_stream_open.next_sample().ravel()
        kmeans_open.step(sample)
        yb=kmeans_open.probabilities
        for t in range(T):
            kmeans_open.step(sample)
            y=kmeans_open.probabilities
            dso.append(np.linalg.norm(y-yb))
            eso.append(kmeans_open.accuracy())
            yb=y
        outer+=1

    outer=0
    while outer<outer_steps and data_stream_closed.has_next_sample():
        sample=data_stream_closed.next_sample().ravel()
        kmeans_closed.step(sample)
        yb=kmeans_closed.probabilities
        ub=sample.copy()
        for t in range(T):
            ud=controller.control(yb,ub).ravel()
            u=sample+ud
            kmeans_closed.step(u)
            y=kmeans_closed.probabilities
            dsc.append(np.linalg.norm(y-yb))
            esc.append(kmeans_closed.accuracy())
            ub=u
            yb=y
        outer+=1
    
    plt.figure(figsize=(6,4))
    plt.plot(dso,label="Without control")
    plt.plot(dsc,label="With control")
    plt.xlabel("Iteration")
    plt.ylabel("||y-yb||")
    plt.legend()
    plt.grid()
    

    plt.figure(figsize=(6,4))
    plt.plot(eso,label="Without control")
    plt.plot(esc,label="With control")
    plt.xlabel("Iteration")
    plt.ylabel("ACC")
    plt.legend()
    plt.grid()
    plt.show()
    