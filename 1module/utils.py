import numpy as np
import scipy as sp
import scipy.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl


def print_center_text(s):
    count = 0
    for i in s:
        count += 1
    pad = (int)((79-count)/2.0)
    print(' '*pad+s)


def step_func(x):
    return 0.5*(1+np.sign(x))


def harmonic_potential(x, omega, D):
    pot = (0.5*(omega**2)*(x**2))+D
    for i in range(len(pot)):
        if pot[i] > 0:
            pot[i] = 0
    return pot


def diagonalize_hamiltonian(Hamiltonian):
    return spla.eigh(Hamiltonian)


def finite_well_plot(E,V,xvec,steps,n,U):
    
    #V = np.multiply(np.conj(V),V) #<- for probability density
    V_new,ScaleFactor,U_new,n=finite_well_plot_scaling(E,V,xvec,U,n,steps)
    # create the figure
    f=plt.figure()
    # add plot to the figure
    ax=f.add_subplot(111)
    # plot potential
    ax.plot(xvec,U_new,c='lightslategray')
    # find appropriate x limits and set x limit
    MinX=0
    MaxX=len(xvec)-1
    while U_new[MinX]==0:
        MinX=MinX+1
    while U_new[MaxX]==0:
        MaxX=MaxX-1
    for m in range(n):
        V_old=V_new[MinX+1,m]
        while(np.abs(V_old - V_new[MinX,m])>1e-6 and MinX>0):
            V_old=V_new[MinX,m]
            MinX=MinX-1
        V_old=V_new[MaxX-1,m]
        while(np.abs(V_old - V_new[MaxX,m])>1e-6 and MaxX<len(xvec)-1):
            V_old=V_new[MaxX,m]
            MaxX=MaxX+1
    plt.xlim(xvec[MinX],xvec[MaxX])
    # find appropriate y limits and set y limit
    if(np.max(V_new)>0):
        if(np.min(V_new)>np.min(U_new)):
            plt.ylim(1.05*np.min(U_new),np.max(V_new)+abs(0.05*np.min(U_new)))
        else:
            plt.ylim(1.05*np.min(V_new),np.max(V_new)+abs(0.05*np.min(U_new)))
    else:
        if(np.min(V_new)>np.min(U_new)):
            plt.ylim(1.05*np.min(U_new),np.max(U_new)+abs(0.05*np.min(U_new)))
        else:
            plt.ylim(1.05*np.min(V_new),np.max(U_new)+abs(0.05*np.min(U_new)))
            # plot wave functions
    for i in np.arange(n-1, -1, -1):
        color = mpl.cm.jet_r((i)/(float)(n), 1)
        wavefunc = ax.plot(xvec, V_new[0:steps, i], c=color, label='E(a.u.)={}'.format(np.round(E[i]*1000)/1000.0))
        ax.axhline(y=V_new[0, i], xmin=-10, xmax=10, c=color, ls='--')
    # set x label
    plt.xlabel('Width of Well / (a.u.)')
    # set y label
    plt.ylabel('Energy / (a.u.)')
    # modify tick marks
    ax.set_yticklabels(np.round(ax.yaxis.get_ticklocs()*ScaleFactor))
    # add plot legend
    L = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.7*box.width, box.height])
    plt.show()

def finite_well_plot_scaling(E,V,xvec,U,n,steps):
    # scale the wave functions
    order=np.argsort(E)
    Converged=False
    while(Converged is False):
        E_copy=E[0:n]
        V_copy=V[0:steps,order]
        V_copy=V[0:steps,0:n]
        max_E_diff = E_copy[n-1] - E_copy[0]
        found_step = False
        step = 1
        while found_step is False:
            if(E_copy[step]-E_copy[0]<0.2):
                step+=1
            else:
                found_step = True
        ScaleFactorStep=0.05
        ScaleFactor=1.00
        Overlap=1
        while(Overlap==1):
            for i in range(0,n,step):
                MaxV2=np.max(V_copy[0:steps,i])+E_copy[i]/ScaleFactor
                MinV2=np.min(V_copy[0:steps,i])+E_copy[i]/ScaleFactor
                MaxV1=np.max(V_copy[0:steps,i-step])+E_copy[i-step]/ScaleFactor
                if((MaxV2-MinV2)<(np.abs(MinV2-MaxV1)*10)):
                    Overlap=1
                else:
                    Overlap=0
                    break
            ScaleFactor=ScaleFactor+ScaleFactorStep
        V_copy_new=(E_copy/ScaleFactor)+V_copy
        if np.max(V_copy_new[n])>0:
            Converged=True
            n=n-1
        else:
            n=n+1
        V_copy_old=V_copy_new
    V_new=V_copy_old
    U_new=U/ScaleFactor
    return V_new,ScaleFactor,U_new,n


def output(input_fields, input_values, E, n):
    print("")
    print('*'*79)
    print_center_text('Schrodinger Solver Output')
    print('*'*79)
    print("")
    print("\t\tInput:")
    for i, j in zip(input_fields, input_values):
        print_center_text(str(i)+' : '+str(j))
    print("")
    print("\t\t{} lowest Bound States:".format(n))
    count = 0
    for i in range(n):
        print_center_text('E({})='.format(i)+str(E[i]))
    print('*'*79)
    print("")