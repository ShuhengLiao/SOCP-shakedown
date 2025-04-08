from fenics import *
import matplotlib.pyplot as plt
from utils import *
from mosek.fusion import *
import mosek.fusion.pythonic
from scipy.sparse import block_diag
from SD_problem import *

# -------------------------------
# Mesh and Function Spaces Setup
# -------------------------------
mesh_file = "../mesh/plate1.inp"
mesh, pvmesh = read_mesh(mesh_file, get_pvmesh=True)
x = SpatialCoordinate(mesh)
metadata = {"quadrature_degree": 1, "quadrature_scheme": "default"}
dxm = dx(metadata=metadata)

U = VectorFunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)
V_strain = VectorFunctionSpace(mesh, "DG", 0, dim=6)

# ------------------
# Material Properties
# ------------------
rho = 2780.
E = 72.4e9
nu = 0.33
yield_stress = 345e6
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# -------------------------
# Boundary Condition Helpers
# -------------------------
def bot(x, on_boundary): return on_boundary and near(x[1], 0.0)
def left(x, on_boundary): return on_boundary and near(x[0], 0.0)
def back(x, on_boundary): return on_boundary and near(x[2], 0.0)

U_bc1 = DirichletBC(U.sub(1), 0.0, bot)
U_bc2 = DirichletBC(U.sub(0), 0.0, left)
U_bc3 = DirichletBC(U.sub(2), 0.0, back)
Mech_BC = [U_bc1, U_bc2, U_bc3]

# ---------------------
# Stress-Strain Methods
# ---------------------
def epsilon(u): return sym(grad(u))
def sigma(u): return lambda_ * div(u) * Identity(3) + 2 * mu * epsilon(u)

def output_stress(sig):
    return np.array([
        project(sig[0, 0], P0).vector(),
        project(sig[1, 1], P0).vector(),
        project(sig[2, 2], P0).vector(),
        project(sig[0, 1], P0).vector(),
        project(sig[0, 2], P0).vector(),
        project(sig[1, 2], P0).vector()
    ])

# ----------------
# Load Boundaries
# ----------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

right = AutoSubDomain(lambda x: near(x[0], mesh.coordinates()[:, 0].max()))
right.mark(boundaries, 1)

top = AutoSubDomain(lambda x: near(x[1], mesh.coordinates()[:, 1].max()))
top.mark(boundaries, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ------------------
# Variational Problem
# ------------------
du = TrialFunction(U)
v = TestFunction(U)
a = inner(sigma(du), epsilon(v)) * dx

def run(p1_value, p2_value):
    p1 = Constant((p1_value, 0.0, 0.0))
    p2 = Constant((0.0, p2_value, 0.0))
    L = inner(p1, v) * ds(1) + inner(p2, v) * ds(2)
    u = Function(U, name="Total displacement")
    solve(a == L, u, Mech_BC)
    return output_stress(sigma(u))

# -----------------------------
# Strain Operator Matrix (H)
# -----------------------------
def apply_BC_to_H(H):
    dirich_dof_ind = []
    for bc in Mech_BC:
        dirich_dof_ind += list(bc.get_boundary_values().keys())
    dirich_dof_ind.sort()
    return delete_from_csr(H, dirich_dof_ind)

def get_H():
    u_ = TrialFunction(U)
    v = TestFunction(V_strain)

    def eps(v):
        e = sym(grad(v))
        return as_vector([e[0, 0], e[1, 1], e[2, 2], e[0, 1], e[0, 2], e[1, 2]])

    B_form = inner(eps(u_), v) * dxm
    B = to_scipy_csr(assemble(B_form))
    H = B.T.tocsr()
    return apply_BC_to_H(H)

# ------------------
# Optimization Setup
# ------------------
N_Gauss = 1
Ni = 4
Nj = mesh.cells().shape[0] * N_Gauss
H = get_H()

args = {'Ni': Ni, 'Nj': Nj, 'H': H}
prob = SD_problem(args)
Y = np.ones(Nj) * yield_stress
prob.update_Q_matrix(Y)
prob.update_A()

# -----------------
# Load Case Testing
# -----------------
p1_list = [1e8, 1e8, 1e8, 0.5e8, 0.0]
p2_list = [0.0, 0.5e8, 1e8, 1e8, 1e8]

r_list = []
r_e_list = []

for p1, p2 in zip(p1_list, p2_list):
    sig1 = run(p1, 0)
    sig2 = run(0, p2)
    sig3 = run(p1, p2)
    sig4 = run(0, 0)
    sig = np.vstack((sig1.flatten('F'), sig2.flatten('F'), sig3.flatten('F'), sig4.flatten('F')))

    prob.update_c(sig)
    r = prob.optimize_r()
    r_e = prob.optimize_r_EL()

    r_list.append(r)
    r_e_list.append(r_e)

# ----------------------
# Plot Normalized Results
# ----------------------
plt.figure()
p1 = np.array(p1_list)
p2 = np.array(p2_list)
r = np.array(r_list)
r_e = np.array(r_e_list)

plt.plot(p1 / r_e / yield_stress, p2 / r_e / yield_stress, '-xb')
plt.plot(p1 / r / yield_stress, p2 / r / yield_stress, '-or')
plt.xlabel('$P_1/\sigma_Y$')
plt.ylabel('$P_2/\sigma_Y$')
plt.axis('equal')
plt.legend(['Shakedown limit', 'Elastic limit'])
plt.savefig('plate_with_hole.pdf')


