import numpy as np
class RLS:
    def __init__(self,ny,nx):
        self.ny=ny;self.nx=nx;
        self.x=np.zeros((nx,1),dtype=float)
        self.P=np.eye(nx,dtype=float)*1e6
    
    def update(self,H,y):
        H=np.asarray(H,dtype=float).reshape(self.ny,self.nx)
        y=np.asarray(y,dtype=float).reshape(self.ny,1)
        Iy=np.eye(self.ny,dtype=float)
        Ix=np.eye(self.nx,dtype=float)
        P=self.P
        x=self.x
        yhat=H@x
        e=y-yhat
        R=Iy+H@P@H.T
        Rinv=np.linalg.inv(R)
        K=P@H.T@Rinv
        self.x=x+K@e
        self.P=(Ix-K@H)@P
        self.P=(self.P+self.P.T)/2
    def __repr__(self):       
        return f"RLS:\nState={self.x.flatten()}\nCovariance=\n{self.P}"                
        