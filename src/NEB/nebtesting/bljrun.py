import numpy as np

import NEB.NEB as NEB
import basinhopping
from quench import quench
#from potentials.lj import LJ
from mindist.minpermdist_stochastic import minPermDistStochastic as minpermdist
from printing.print_atoms_xyz import printAtomsXYZ


def printpath(fout, coordslist, atomtypes = ["LA"]):
    nimages = len(coordslist[:,0])
    for i in range(nimages):
        printAtomsXYZ(fout, coordslist[i,:], atom_type=atomtypes)

def printpath_EoS(fout, coordslist, getEnergy):
    nimages = len(coordslist[:,0])
    S = 0.
    for i in range(nimages-1):
        xyz1 = neb.coords[i,:]
        xyz2 = neb.coords[i+1,:]
        dist = np.linalg.norm(xyz1 - xyz2)
        E = getEnergy(xyz1)
        fout.write( str(S) + " " + str(E) + "\n")
        S += dist
    xyz = neb.coords[-1,:]
    E = getEnergy(xyz)
    fout.write( str(S) + " " + str(E) + "\n")


from potentials.ljpshift import LJpshift
natoms = 17
ntypea = int(natoms*.8)
lj = LJpshift(natoms, ntypea)
permlist = [range(ntypea), range(ntypea, natoms)]


X1 = np.random.uniform(-1,1,[natoms*3])*(float(natoms))**(1./3) * .8
ret = quench( X1, lj.getEnergyGradient)
X1 = ret[0]
X2 = np.random.uniform(-1,1,[natoms*3])*(float(natoms))**(1./3) * .8
ret = quench( X2, lj.getEnergyGradient)
X2 = ret[0]

dist, X1, X2 = minpermdist( X1, X2, niter = 100, permlist=permlist )
distf = np.linalg.norm(X1 - X2)
print "dist returned        ", dist
print "dist from structures ", distf

#X1 = np.array( [ 0., 0., 0., 1., 0., 0., 0., 0., 1.,] )
#X2 = np.array( [ 0., 0., 0., 1., 0., 0., 0., 1., 0.,] )
import copy
X1i = copy.copy(X1)
X2i = copy.copy(X2)

atomtypes = ["L" for i in range(ntypea)] #for printing binary systems
for i in range(natoms - ntypea):
    atomtypes.append("O")
print atomtypes
print "setting up path" 
neb = NEB.NEB(X1, X2, lj, k = 100. )
print "saving intial path to path.init.xyz"
with open("path.init.xyz", "w") as fout:
    printpath(fout, neb.coords, atomtypes)


npaths = len(neb.coords[:,0])
print "intial path distances", npaths
for i in range(npaths-1):
    xyz1 = neb.coords[i,:]
    xyz2 = neb.coords[i+1,:]
    dist = np.linalg.norm(xyz1 - xyz2)
    print "   ", dist
with open("path.init.EoS", "w") as fout:
    printpath_EoS(fout, neb.coords, lj.getEnergy) 


print "optimizing path"
#neb.optimize(neb.bfgs_quench)
neb.optimize()

print "saving final path to path.final.xyz"
with open("path.final.xyz", "w") as fout:
    printpath(fout, neb.coords, atomtypes)
    
print "final path distances", npaths
for i in range(npaths-1):
    xyz1 = neb.coords[i,:]
    xyz2 = neb.coords[i+1,:]
    dist = np.linalg.norm(xyz1 - xyz2)
    print "   ", dist
with open("path.final.EoS", "w") as fout:
    printpath_EoS(fout, neb.coords, lj.getEnergy) 

