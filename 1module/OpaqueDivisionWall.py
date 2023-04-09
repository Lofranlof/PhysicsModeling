from utils import *

# atomic units
hbar = 1.0
m = 1.0
# set precision of numerical approximation
steps = 2000

# set depths and widths of wells and the well separation
W = 1
D = 25.0
B = 0.05
Vinf = 20
Opacity = 1/2

# set length variable for xvec
A = 2.0*((2*W)+B)
# divide by two so a separation from -B to B is of input size
B = B/2.0
# create x-vector from -A to A
xvec = np.linspace(-A, A, steps, dtype=np.float_)
# get step size
h = xvec[1]-xvec[0]
# create the potential from step function
U = -D* (step_func(xvec+W+B)-step_func(xvec+B) + step_func(xvec-B)-step_func(xvec-W-B)) + Vinf*(1-0)*(step_func(xvec+B) - step_func(xvec-B))
# create Laplacian via 3-point finite-difference method
Laplacian = (-2.0*np.diag(np.ones(steps))+np.diag(np.ones(steps-1), 1)
             + np.diag(np.ones(steps-1), -1))/(float)(h**2)
# create the Hamiltonian
Hamiltonian = np.zeros((steps, steps))
[i, j] = np.indices(Hamiltonian.shape)
Hamiltonian[i == j] = U
Hamiltonian += (-0.5)*((hbar**2)/m)*Laplacian
# diagonalize the Hamiltonian yielding the wavefunctions and energies
E, V = diagonalize_hamiltonian(Hamiltonian)
# determine number of energy levels to plot (n)
n = 0
while E[n] < 0:
    n += 1
# print output
output(['Well Widths', 'Well Depths', 'Well Separation'], [W, D, B*2], E, n)
# create plot
finite_well_plot(E, V, xvec, steps, n, U)