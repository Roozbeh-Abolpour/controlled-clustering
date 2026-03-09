import numpy as np
import cvxpy as cp
from controlled_clustering.identification import arx as arx_module

class MPC:
    def __init__(self,ny,my,nu,mu,N,umin,umax):
        self.ny=ny;self.nu=nu;self.my=my;self.mu=mu;self.N=N;
        self.umin=np.asarray(umin,dtype=float).reshape(nu,1)
        self.umax=np.asarray(umax,dtype=float).reshape(nu,1)
        self.arx=arx_module.ARX(ny,my,nu,mu)
    
    def control(self,yb,ub):
        ny=self.ny;nu=self.nu;my=self.my;mu=self.mu;N=self.N;
        self.arx.update(yb,ub)
        Ud=self.arx.Ud
        Yd=self.arx.Yd
        As,Bs=self.arx.arxmats()
        Uc=cp.Variable((nu*N,1))
        Yc=cp.Variable((ny*N,1))
        
        Constraints=[]
        for k in range(N):
            yp=0
            for j in range(my):
                if k-j-1>=0:
                    yp+=As[j]@Yc[(k-j-1)*ny:(k-j)*ny]
                else:
                    yp+=As[j]@Yd[(j-k)*ny:(j-k+1)*ny]
            for j in range(mu):
                if k-j-1>=0:
                    yp+=Bs[j]@Uc[(k-j-1)*nu:(k-j)*nu]
                else:
                    yp+=Bs[j]@Ud[(j-k)*nu:(j-k+1)*nu]
            Constraints.append(Yc[k*ny:(k+1)*ny]==yp)
        
        Constraints.append(Uc>=np.kron(np.ones((N,1),dtype=float),self.umin))
        Constraints.append(Uc<=np.kron(np.ones((N,1),dtype=float),self.umax))
        
        cost=0
        for k in range(N):
            if k==0:
                cost+=cp.sum_squares(Yc[k*ny:(k+1)*ny]-yb)
            else:         
                cost+=cp.sum_squares(Yc[k*ny:(k+1)*ny]-Yc[(k-1)*ny:k*ny])
        
        
        objective=cp.Minimize(cost)
        problem=cp.Problem(objective,Constraints)
        problem.solve()
        return Uc.value[:nu,:]