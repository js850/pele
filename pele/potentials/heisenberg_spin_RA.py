import numpy as np
from numpy import sin, cos
from copy import copy
import networkx as nx

from pele.potentials import BasePotential
import pele.utils.rotations as rotations
from pele.potentials.heisenberg_spin import make3dVector,  make2dVector, coords2ToCoords3, coords3ToCoords2, grad3ToGrad2



__all__ = ["HeisenbergModelRA"]



class HeisenbergModelRA(BasePotential):
    """
    The classical Heisenberg Model of 3d spins on a lattice with random anisotropy.
    
    The Hamiltonian is
        
    H = - sum_ij J_ij dot( s_i, s_j )  - sum_i dot( h_i, s_i )**2
    
    where h_i are quenched random variables.  (h_i is a vector)
    """
    def __init__(self, dim = [4, 4], field_disorder = 1.):
        """
        dim is an array giving the dimensions of the lattice
        
        phi is the magnitude of the randomness in the fields
        """
        self.dim = copy(dim)
        self.nspins = np.prod(dim)
        
        self.G = nx.grid_graph(dim, periodic=True)
        
        self.fields = np.zeros([self.nspins, 3])
                
        self.indices = dict()
        i = 0
        for node in self.G.nodes():
            self.indices[node] = i
            self.fields[i,:] = rotations.vec_random() * \
                np.sqrt(field_disorder) #np.random.uniform(0, field_disorder, [3])
            i += 1 


    
        
    def getEnergy(self, coords):
        """
        coords is a list of (theta, phi) spherical coordinates of the spins
        where phi is the azimuthal angle (angle to the z axis) 
        """
        coords3 = coords2ToCoords3( coords )
            
        E = 0.
        for edge in self.G.edges():
            u = self.indices[edge[0]]
            v = self.indices[edge[1]]
            E -= np.dot( coords3[u,:], coords3[v,:] )
        
        Efields = - np.sum( np.sum( self.fields * coords3, axis=1 )**2 )
        
        #print "EJ EH", E, Efields
        return E + Efields
        
    def getEnergyGradient(self, coords):
        """
        coords is a list of (theta, phi) spherical coordinates of the spins
        where phi is the azimuthal angle (angle to the z axis) 
        """
        coords3 = coords2ToCoords3( coords )
        coords2 = coords
            
        E = 0.
        grad3 = np.zeros( [self.nspins, 3] )
        for edge in self.G.edges():
            u = self.indices[edge[0]]
            v = self.indices[edge[1]]
            E -= np.dot( coords3[u,:], coords3[v,:] )
            
            grad3[u,:] -= coords3[v,:]
            grad3[v,:] -= coords3[u,:]
        
        vdotf = np.sum( self.fields * coords3, axis=1 )
        Efields = - np.sum( np.sum( self.fields * coords3, axis=1 )**2 )

        grad3 -= 2.* self.fields * vdotf[:, np.newaxis]
        
        #for i in range(self.nspins):
        #    grad3[i,:] /= np.linalg.norm( grad3[i,:] )
        
        grad2 = grad3ToGrad2(coords2, grad3)
        grad2 = np.reshape(grad2, self.nspins*2)
        
        #print "EJ EH", E, Efields
        return E + Efields, grad2



def test_basin_hopping(pot, angles):
    from pele.basinhopping import BasinHopping
    from pele.takestep.displace import RandomDisplacement
    from pele.takestep.adaptive import AdaptiveStepsize
    
    takestep = RandomDisplacement(stepsize = np.pi/4)
    takestepa = AdaptiveStepsize(takestep, frequency = 10)
    
    bh = BasinHopping( angles, pot, takestepa, temperature = 1.01)
    bh.run(200)

def test():
    pi = np.pi
    L = 4
    nspins = L**2
    
    #phases = np.zeros(nspins)
    pot = HeisenbergModelRA( dim = [L,L], field_disorder = 2. ) #, phases=phases)
    
    coords = np.zeros([nspins, 2])
    for i in range(nspins): 
        vec = rotations.vec_random()
        coords[i,:] = make2dVector(vec)
    coords = np.reshape(coords, [nspins*2])
    if False:
        normfields = np.copy(pot.fields)
        for i in range(nspins): normfields[i,:] /= np.linalg.norm(normfields[i,:])
        coords = coords3ToCoords2( np.reshape(normfields, [nspins*3] ) )
        coords  = np.reshape(coords, nspins*2)
    #print np.shape(coords)
    coordsinit = np.copy(coords)
    
    #print "fields", pot.fields
    print coords
    
    if False:
        coords3 = coords2ToCoords3(coords)
        coords2 = coords3ToCoords2(coords3)
        print np.reshape(coords, [nspins,2])
        print coords2
        coords3new = coords2ToCoords3(coords2)
        print coords3
        print coords3new

    e = pot.getEnergy(coords)
    print "energy ", e
    if True:
        print "numerical gradient"
        ret = pot.getEnergyGradientNumerical(coords)
        print ret[1]
        if True:
            print "analytical gradient"
            ret2 = pot.getEnergyGradient(coords)
            print ret2[1]
            print ret[0]
            print ret2[0]
            #print "ratio"
            #print ret2[1] / ret[1]
            #print "inverse sin"
            #print 1./sin(coords)
            #print cos(coords)
    
    print "try a quench"
    from pele.optimize import lbfgs_cpp
    ret = lbfgs_cpp(coords, pot)
    
    print "quenched e = ", ret.energy, "funcalls", ret.nfev
    print ret.coords
    with open("out.spins", "w") as fout:
        s = coords2ToCoords3( ret.coords )
        h = pot.fields
        c = coords2ToCoords3( coordsinit )
        for node in pot.G.nodes():
            i = pot.indices[node]
            fout.write( "%g %g %g %g %g %g %g %g %g %g %g\n" % (node[0], node[1], \
                s[i,0], s[i,1], s[i,2], h[i,0], h[i,1], h[i,2], c[i,0], c[i,1], c[i,2] ) )
    
    coords3 = coords2ToCoords3( ret.coords )
    m = np.linalg.norm( coords3.sum(0) ) / nspins
    print "magnetization after quench", m
    
    test_basin_hopping(pot, coords)
    
if __name__ == "__main__":
    test()
