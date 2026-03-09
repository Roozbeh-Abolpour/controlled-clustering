import numpy as np
import controlled_clustering.identification.rls as rls

class ARX:
    def __init__(self,ny,my,nu,mu):      
        self.ny=ny;self.my=my;self.nu=nu;self.mu=mu;
        self.Yd=np.zeros((my*ny,1),dtype=float)
        self.Ud=np.zeros((nu*mu,1),dtype=float)
        self.rls=rls.RLS(ny,ny*ny*my+ny*nu*mu)         

    def arxmats(self):
        theta=self.rls.x
        ny=self.ny;nu=self.nu;my=self.my;mu=self.mu;   
        ind=0
        As=[]
        for i in range(my):
            Ae=theta[ind:ind+ny*ny].reshape(ny,ny)
            ind+=ny*ny
            As.append(Ae)
        Bs=[];    
        for i in range(mu):
            Be=theta[ind:ind+ny*nu].reshape(ny,nu)
            ind+=ny*nu
            Bs.append(Be)
        return As,Bs
    
    def update(self,y,u):
        ny=self.ny;nu=self.nu;my=self.my;mu=self.mu;          

        y=np.asarray(y,dtype=float).reshape(ny,1)
        u=np.asarray(u,dtype=float).reshape(nu,1)
        Ud=self.Ud;Yd=self.Yd;
        Ud=np.vstack((u,Ud[:-nu]))
        Yd=np.vstack((y,Yd[:-ny]))        

        Iy=np.eye(ny,dtype=float)                
        H=np.zeros((ny,ny*ny*my+ny*nu*mu),dtype=float)
        ind=0        
        for i in range(my):
            yd=Yd[i*ny:(i+1)*ny]
            H[:,ind:ind+ny*ny]=np.kron(yd.T,Iy)
            ind+=ny*ny
        for i in range(mu):
            ud=Ud[i*nu:(i+1)*nu]
            H[:,ind:ind+ny*nu]=np.kron(ud.T,Iy)
            ind+=ny*nu
        self.rls.update(H,y)
        self.Ud=Ud;self.Yd=Yd;
