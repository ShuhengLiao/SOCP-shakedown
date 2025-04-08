import numpy as np
from scipy.sparse import block_diag
from mosek.fusion import *
from utils import *
import sys

class SD_problem:
    def __init__(self, args, prob_type = '3D'):
        self.type = prob_type
        self.Ni = args['Ni']
        self.Nj = args['Nj']
        self.H = args['H']
        if self.type == '3D':
            self.Ntau = 5
            self.Nsig = 6
        if self.type == 'Axisymmetric':
            self.Ntau = 3
            self.Nsig = 4

    def update_Q_matrix(self,Y):
        if self.type == '3D':
            Q_local = np.array([[2**0.5/2., 2**0.5/2., 2**0.5/2., 0.,0.,0.],
                       [1/2.,-1, 1/2., 0., 0.,0.],
                       [-3**0.5/2.,0.,3**0.5/2., 0., 0.,0.],
                       [0.,0.,0.,3**0.5,0.,0.],
                       [0.,0.,0.,0.,3**0.5,0.],
                       [0.,0.,0.,0.,0.,3**0.5]])
        if self.type == 'Axisymmetric':
            Q_local = np.array([[2**0.5/2., 2**0.5/2., 2**0.5/2., 0.],
                       [1/2.,-1, 1/2., 0.],
                       [-3**0.5/2.,0.,3**0.5/2., 0.],
                       [0.,0.,0.,3**0.5]])
        Q_inv_local = np.linalg.inv(Q_local)
        self.Q = block_diag([Q_local/Y[j] for j in range(self.Nj)])
        self.Q_inv = block_diag([Q_inv_local*Y[j] for j in range(self.Nj)])
        
    def update_A(self):
        self.A = self.H@self.Q_inv
        Ap = self.A[:,::self.Nsig]
        At= delete_from_csr(self.A,row_indices=[],col_indices=list(np.arange(0,self.Nj*self.Nsig,self.Nsig)))
        self.At = to_mosek_sparse(At)
        self.Ap = to_mosek_sparse(Ap)
        
    def update_c(self,sig):
        self.f_ext = self.A@self.Q@sig[0]
        c = np.array([self.Q@sig[i] for i in range(self.Ni)])
        self.ct = np.delete(c,np.arange(0,self.Nj*self.Nsig,self.Nsig),axis=1)
    
#     def optimize_r_EL(self):
#         M = Model('cqo1')
#         r = M.variable('r', 1, Domain.greaterThan(0.0))

#         ct = Expr.constTerm(self.ct.reshape(-1,self.Ntau))
#         cone_expr = Expr.hstack(Var.repeat(r,self.Ni*self.Nj),ct)
#         M.constraint("Cone", cone_expr, Domain.inQCone())

#         M.objective("obj", ObjectiveSense.Minimize, r)
#         M.setLogHandler(sys.stdout)
#         # Solve the problem
#         M.solve()
        
#         return r.level()
    
    
    def optimize_r_EL(self):
        ct = self.ct.reshape(self.Ni,self.Nj,self.Ntau)
        r = ((ct[:,:,0]**2 + ct[:,:,1]**2 + ct[:,:,2]**2)**0.5).max()
        return r
    
    def optimize_r(self):
        M = Model('cqo1')
        r = M.variable('r', 1, Domain.greaterThan(0.0))
        y_t = M.variable('y_t', [self.Ni*self.Nj,self.Ntau], Domain.unbounded())
        y_1p = M.variable('y_p', self.Nj, Domain.unbounded())

        cone_expr = Var.hstack(Var.repeat(r,self.Ni*self.Nj),y_t)
        M.constraint("Cone", cone_expr, Domain.inQCone())
        M.setSolverParam("intpntCoTolDfeas", 1.0e-6)
        for i in range(1,self.Ni):
            lhs = (y_t[self.Nj*i:self.Nj*(i+1),:] - y_t[:self.Nj,:]) 
            rhs = (self.ct[i] - self.ct[0]).reshape(-1,self.Ntau)
            M.constraint("EQ_{}".format(i), lhs==rhs)

#         M.constraint("EQ", self.At@y_t[0:self.Nj,:].reshape(self.Nj*self.Ntau) + (self.Ap@y_1p) == self.f_ext)
        
        M.constraint("EQ", self.At@y_t[0:self.Nj,:].reshape(self.Nj*self.Ntau) + (self.Ap@y_1p) == self.f_ext)

        M.objective("obj", ObjectiveSense.Minimize, r)
        M.setLogHandler(sys.stdout)
        # Solve the problem
        M.solve()
        return r.level()[0]
        
        
  