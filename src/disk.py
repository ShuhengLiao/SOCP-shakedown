from dolfin import *
import meshio
import pyvista as pv
import numpy as np
from utils import *
from materials import *
parameters["form_compiler"]["representation"] = 'quadrature'
import warnings
import sys
from SD_problem import *
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
from mosek.fusion import *
import mosek.fusion.pythonic

disk_name = 'disk_24'

mesh_file = "../mesh/disk_fine.xdmf" # .inp or .xdmf
mesh,pvmesh = read_mesh(mesh_file,get_pvmesh=True)
metadata = {"quadrature_degree": 1, "quadrature_scheme": "default"}
dxm = dx(metadata=metadata)


# ####### manually define a gradient disk
# # 1. pure IN718
# comp_list = np.linspace(1.,1.,1) # a list defining the composition, e.g., [1,0.9,0.5,0.4,0.3]

# # 2. IN718 - FINVAR
# # comp_list1 = np.linspace(1.,1.,101)
# # comp_list2 = np.linspace(1.,0.,22)[1:-1]
# # comp_list3 = np.linspace(0.,0.,15)
# # comp_list = np.concatenate((comp_list1,comp_list2,comp_list3))

# properties = get_properties(mesh,comp_list)


##### read the property file
# mesh = scale_mesh(mesh, 0.1745590688908787, 0.01060179185682686, mesh.coordinates()[:,1].max())
comp_list = np.arange(100)
comp2prop = get_composition2property_from_csv('../disk_properties/{}_properties.csv'.format(disk_name))
properties = get_properties(mesh,comp_list,comp2prop)


t_rise = 1. # time to heat to the max temp.
t_heatdwell = 20. # heating dwell time
t_fall = 3. # time to cool to the ambient temp
t_cooldwell = 600. # cooling dwell time
time_intervals = [t_rise, t_heatdwell, t_fall, t_cooldwell]
step_list = [t_rise/2., t_heatdwell/20., t_fall/2., t_cooldwell/5.]
t_list = get_time_list_alt(time_intervals,step_list)

# material properties
rho,cp,kappa,E,sig0,nu,alpha_V = properties
lmbda = E*nu/(1+nu)/(1-2*nu)
mu = E/2./(1+nu)
Et = E/1e5  # tangent modulus - any better method for perfect plasticity?
H = E*Et/(E-Et)  # hardening modulus
Y = properties[4].vector()[:]

# DEFINE THERMAL PROBLEM
U = FunctionSpace(mesh, "CG", 1)  # Temperature space
v = TestFunction(U)
x = SpatialCoordinate(mesh) # Coords
T_initial = Constant(0.)
T_pre = interpolate(T_initial, U) # Temp. at last time step
T_crt = TrialFunction(U)
dt = Constant(1.)
F_thermal = (rho*cp*(T_crt-T_pre)/dt)*v*x[0]*dx + kappa*dot(grad(T_crt),grad(v))*x[0]*dx
a_thermal, L_thermal = lhs(F_thermal), rhs(F_thermal)

# DEFINE MECH PROBLEM
V = VectorFunctionSpace(mesh, "CG", 1)
We = VectorElement("Quadrature", mesh.ufl_cell(), degree=1, dim=4, quad_scheme='default')
W = FunctionSpace(mesh, We)
W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=1, quad_scheme='default')
W0 = FunctionSpace(mesh, W0e)

# V = VectorFunctionSpace(mesh, "CG", 1, dim=3)
V_strain = VectorFunctionSpace(mesh, "DG", 0, dim=4)

def eps(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0]/x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))

def sigma(v, dT):
    return (lmbda*tr(eps(v))- alpha_V*(3*lmbda+2*mu)*dT)*Identity(3) + 2.0*mu*eps(v)

def output_stress(sig):
    sig11 = project(sig[0,0],FunctionSpace(mesh, "DG", 0)).vector()
    sig22 = project(sig[1,1],FunctionSpace(mesh, "DG", 0)).vector()
    sig33 = project(sig[2,2],FunctionSpace(mesh, "DG", 0)).vector()
    sig13 = project(sig[0,2],FunctionSpace(mesh, "DG", 0)).vector()
    return np.array([sig11,sig22,sig33,sig13])

## DEFINE BCs
boundary_left,boundary_right,boundary_bot,_ = get_boundary(mesh)

T_L = Constant(0.)
T_R = Constant(0.)
T_bc_L = DirichletBC(U, T_L, boundary_left)
T_bc_R = DirichletBC(U, T_R, boundary_right)
Thermal_BC = [T_bc_L,T_bc_R]

U_bc_B = DirichletBC(V.sub(1), 0., boundary_bot)
U_bc_L = DirichletBC(V.sub(0), 0., boundary_left)
Mech_BC = [U_bc_B,U_bc_L]


def run_simulation(load,omega,t_list):
    def F_int(v):
        return rho*omega**2*x[0]*v[0]

    dT = Function(U)
    v_ = TrialFunction(V)
    u_ = TestFunction(V)
    Wint = inner(sigma(v_, dT), eps(u_))*x[0]*dxm - F_int(u_)*x[0]*dxm
    a_m, L_m = lhs(Wint),rhs(Wint)

    T_crt = Function(U, name="Temperature")
    u = Function(V, name="Total displacement")

    sig_list = []

    dT.assign(interpolate(Constant(0.), U))
    solve(a_m == L_m, u, Mech_BC)
    sig = sigma(u,dT)
    sig_list.append(output_stress(sig))

    for n in range(len(t_list)-1): 
        dt.assign(t_list[n+1]-t_list[n])
        T_R.assign(load(t_list[n+1]))
        solve(a_thermal == L_thermal, T_crt, Thermal_BC)
        T_pre.assign(T_crt)

        dT.assign(T_crt-interpolate(T_initial, U))
        solve(a_m == L_m, u, Mech_BC)
        sig = sigma(u,dT)
        sig_list.append(output_stress(sig))
    return sig_list

def apply_BC_to_H(H):
    dirich_dof_ind = []
    for bc in Mech_BC:
        dirich_dof_ind += list(bc.get_boundary_values().keys())
    dirich_dof_ind.sort()
    H_bc = delete_from_csr(H, dirich_dof_ind)
    return H_bc

def get_H():
    u_ = TrialFunction(V)
    v = TestFunction(V_strain)
    
    def eps(v):
        return as_vector([v[0].dx(0), v[0]/x[0], v[1].dx(1), (v[1].dx(0)+v[0].dx(1))/2.])

    b = inner(eps(u_),v)*x[0]*dxm
    B = to_scipy_csr(assemble(b))
    
    dirich_dof_ind = []
    for bc in Mech_BC:
        dirich_dof_ind += list(bc.get_boundary_values().keys())
    dirich_dof_ind.sort()
    
    H = B.T.tocsr()

    return apply_BC_to_H(H)


def get_sig(omega,T_load):
    load = get_loadfunc(T_load,t_rise,t_heatdwell,t_fall,t_cooldwell)
    sig = run_simulation(load,omega,t_list)
    return np.array(sig)

def run(omega0=100,T0=50,scale=10.,num_dir=19):
    N_Gauss = 1
    Ni = t_list.shape[0]
    Nj = mesh.cells().shape[0]*N_Gauss
    H = get_H()

    args = {'Ni':Ni,'Nj':Nj,'H':H}

    prob = SD_problem(args,'Axisymmetric')

    prob.update_Q_matrix(Y)
    prob.update_A()
    
    sig = get_sig(omega0,0)
    sig = sig.reshape((sig.shape[0],-1),order='f')
    prob.update_c(sig)
    r = prob.optimize_r_EL()
    omega0 = omega0/r**0.5

    sig = get_sig(1e-5,T0)
    sig = sig.reshape((sig.shape[0],-1),order='f')
    prob.update_c(sig)
    r = prob.optimize_r_EL()
    T0 = T0/r
    
  
    omega0 = omega0/scale**0.5
    T0 = T0/scale

    omega_list_EL = []
    T_list_EL = []

    omega_list_SD = []
    T_list_SD = []

    for theta in np.linspace(0,pi/2,num_dir):
        omega = np.sin(theta)**0.5*omega0+1e-5
        T_load = np.cos(theta)*T0

        sig = get_sig(omega,T_load)
        sig = sig.reshape((sig.shape[0],-1),order='f')


        prob.update_c(sig)
        r = prob.optimize_r_EL()
        T_list_EL.append(T_load/r)
        omega_list_EL.append((omega/r**0.5))

        r = prob.optimize_r()
        T_list_SD.append(T_load/r)
        omega_list_SD.append((omega/r**0.5)) 
        
    return T_list_EL, T_list_SD, omega_list_EL, omega_list_SD

if __name__ == "__main__":
    results = run(scale=20)
    np.save('results/{}_{}s.npy'.format(disk_name,t_heatdwell),results)