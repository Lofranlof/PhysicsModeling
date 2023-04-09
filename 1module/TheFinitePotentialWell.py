from utils import *

# atomic units
hbar=1.0
m=1.0
#set precision of numerical approximation
steps=2000


# set well depth and width
A=2.0 # this value must be between 1 and 15
D=25.0 # this value must be between 20 and 500

# divide by two so a well from -W to W is of input width
W=A/2.0
# create x-vector from -W to W
xvec=np.linspace(-A,A,steps,dtype=np.float_)
# get step size
h=xvec[1]-xvec[0]
# create the potential from step function
U=-D*(step_func(xvec+W)-step_func(xvec-W))
# create Laplacian via 3-point finite-difference method
Laplacian=(-2.0*np.diag(np.ones(steps))+np.diag(np.ones(steps-1),1)\
    +np.diag(np.ones(steps-1),-1))/(float)(h**2)
# create the Hamiltonian
Hamiltonian=np.zeros((steps,steps))
[i,j]=np.indices(Hamiltonian.shape)
Hamiltonian[i==j]=U
Hamiltonian+=(-0.5)*((hbar**2)/m)*Laplacian
# diagonalize the Hamiltonian yielding the wavefunctions and energies
E,V=diagonalize_hamiltonian(Hamiltonian)
# determine number of energy levels to plot (n)
n=0
while E[n]<0:
    n+=1
# print output
output(['Well Width','Well Depth'],[W*2,D],E,n)
# create plot
finite_well_plot(E,V,xvec,steps,n,U)