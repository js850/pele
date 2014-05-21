import numpy as np
import copy
import logging

from pele.optimize import Result
from pele.potentials.potential import BasePotential
from pele.transition_states import findLowestEigenVector, FindLowestEigenVector
from pele.transition_states._dimer_translator import _DimerTranslator
from pele.transition_states._transverse_walker import _TransverseWalker


__all__ = ["findTransitionState", "FindTransitionState"]

logger = logging.getLogger("pele.connect.findTS")

class FindTransitionState(object):
    """
    This class implements the hybrid eigenvector following routine for finding the nearest transition state
    
    ***orthogZeroEigs is system dependent, don't forget to set it***
    
    Parameters
    ----------
    coords : 
        the starting coordinates
    pot : 
        the potential class
    tol : 
        the tolerance for the rms gradient
    nsteps : 
        number of iterations
    eigenvec0 : 
        a guess for the initial lowest eigenvector
    nsteps_tangent1, nsteps_tangent2 : int
        the number of iterations for tangent space minimization before and after
        the eigenvalue is deemed to be converged
    event : callable
        This will be called after each step
    iprint :
        the interval at which to print status messages
    verbosity : int
        how much debugging information to print (only partially implemented)
    orthogZeroEigs : callable
        this function makes a vector orthogonal to the known zero
        eigenvectors

            orthogZeroEigs=0  : default behavior, assume translational and
                                rotational symmetry
            orthogZeroEigs=None : the vector is unchanged

    lowestEigenvectorQuenchParams : dict 
        these parameters are passed to the quench routine for he lowest
        eigenvector search 
    tangentSpaceQuenchParams : dict 
        these parameters are passed quench routine for the minimization in
        the space tabgent to the lowest eigenvector
    max_uphill_step_initial : float
        The initial maximum uphill step along the direction of the lowest 
        eigenvector.  The maximum uphill step is adjusted dynamically using a 
        trust radius
    max_uphill_step : float 
        The maximum value of the maximum uphill step. The maximum uphill step 
        is adjusted using a trust radius, but it can never become larger than 
        this.  
    check_negative : bool
        If True then the sign of the lowest eigenvalue is required to remain negative.
        If the sign becomes positive, then the step is retaken with smaller step size
    demand_initial_negative_vec : bool
        if True, and check_negative is True, then abort if the initial 
        lowest eigenvalue is positive
    negatives_before_check : int
        If the run starts with a positive eigenvalue and demand_initial_negative_vec is False,
        then check to make sure that the eigenvalue is enabled after having had so 
        many negative eigenvalues before.  If check_negative is False this has no affect.
    nfail_max :
        if the lowest eigenvector search fails this many times in a row
        than the algorithm ends
        
    
    Notes
    -----
    
    It is composed of the following steps:
    
        1) Find eigenvector corresponding to the lowest *nonzero*
           eigenvector.  
        
        2) Step uphill in the direction of the lowest eigenvector
        
        3) minimize in the space tangent to the lowest eigenvector
        
    The tolerances for the various steps of this algorithm must be correlated.
    if the tolerance for tangent space search is lower than the total tolerance, 
    then it will never finish
    
    See Also
    --------
    findTransitionState : function wrapper for this class
    findLowestEigenVector : a core algorithm
    pele.landscape.LocalConnect : the class which most often calls this routine
    """
    def __init__(self, coords, pot, tol=1e-4, event=None, nsteps=100, 
                 nfail_max=200, eigenvec0=None, iprint=-1, orthogZeroEigs=0,
                 nsteps_tangent1=10,
                 nsteps_tangent2=100,
                 lowestEigenvectorQuenchParams=dict(),
                 tangentSpaceQuenchParams=dict(), 
                 max_uphill_step=0.5,
                 max_uphill_step_initial=0.2,
                 demand_initial_negative_vec=False,
                 negatives_before_check = 10,
                 verbosity=1,
                 check_negative=False,
                 invert_gradient=False,
                 ):
        self.pot = pot
        self.coords = np.copy(coords)
        self.nfev = 0
        self.tol = tol
        self.nsteps = nsteps
        self.event = event
        self.nfail_max = nfail_max
        self.nfail = 0
        self.eigenvec = eigenvec0
        self.orthogZeroEigs = orthogZeroEigs
        self.iprint = iprint
        self.lowestEigenvectorQuenchParams = lowestEigenvectorQuenchParams
        self.max_uphill_step = max_uphill_step
        self.verbosity = verbosity
        self.tangent_space_quench_params = dict(tangentSpaceQuenchParams.items())
        self.demand_initial_negative_vec = demand_initial_negative_vec    
        self.npositive_max = max(10, self.nsteps / 5)
        self.check_negative = check_negative
        self.invert_gradient = invert_gradient

        
        self.rmsnorm = 1./np.sqrt(float(len(coords)))
        self.oldeigenvec = None

        #set tolerance for the tangent space minimization.  
        #Be sure it is at least as tight as self.tol
        self.tol_tangent = self.tol# * 0.2
        if self.tangent_space_quench_params.has_key("tol"):
            self.tol_tangent = min(self.tol_tangent, 
                                   self.tangent_space_quench_params["tol"])
            del self.tangent_space_quench_params["tol"]
        self.tangent_space_quench_params["tol"] = self.tol
        
        
        self.nsteps_tangent1 = nsteps_tangent1
        self.nsteps_tangent2 = nsteps_tangent2
        
        
        if self.tangent_space_quench_params.has_key("maxstep"):
            self.maxstep_tangent = self.tangent_space_quench_params["maxstep"]
            del self.tangent_space_quench_params["maxstep"]
        else:
            self.maxstep_tangent = 0.1 #this should be determined in a better way
        
        if not self.tangent_space_quench_params.has_key("logger"):
            self.tangent_space_quench_params["logger"] = logging.getLogger("pele.connect.findTS.tangent_space_quench")


        #set some parameters used in finding lowest eigenvector
        #initial guess for Hermitian
#        try:
#            self.H0_leig = self.lowestEigenvectorQuenchParams.pop("H0")
#        except KeyError:
#            self.H0_leig = None
        
        self.reduce_step = 0
        self.step_factor = .1
        self.npositive = 0
        
        self._trust_radius = 2.
        self._max_uphill = max_uphill_step_initial
        self._max_uphill_min = .01
        self._max_uphill_max = max_uphill_step
        
        self._transverse_walker = None
        
        
        
    @classmethod
    def params(cls, obj = None):
        if obj is None:
            obj = FindTransitionState(np.zeros(2), None)
        
        params = dict()
        
        params["tangentSpaceQuenchParams"] = obj.tangent_space_quench_params.copy()
        params["lowestEigenvectorQuenchParams"] = obj.lowestEigenvectorQuenchParams.copy()
        params["tol"] = obj.tol
        params["nsteps"] = obj.nsteps
        params["nfail_max"] = obj.nfail_max
        params["iprint"] = obj.iprint
        params["nsteps_tangent1"]=obj.nsteps_tangent1
        params["nsteps_tangent2"]=obj.nsteps_tangent2
        params["max_uphill_step"]=obj._max_uphill_max
        params["max_uphill_step_initial"]=obj._max_uphill
        params["demand_initial_negative_vec"]=obj.demand_initial_negative_vec
        params["check_negative"]=obj.check_negative
        params["invert_gradient"]=obj.invert_gradient
        params["verbosity"]=obj.verbosity
        
        # event=None, eigenvec0=None, orthogZeroEigs=0,
        return params
    
    def _saveState(self, coords):
        """save the state in order to revert a step"""
        self.saved_coords = np.copy(coords)
        self.saved_eigenvec = np.copy(self.eigenvec)
        self.saved_eigenval = self.eigenval
        self.saved_overlap = self.overlap
#        self.saved_H0_leig = self.H0_leig
        self.saved_energy = self.energy
        self.saved_gradient = self.gradient.copy()
        #self.saved_oldeigenvec = np.copy(self.oldeigenvec)

    def _resetState(self):
        """restore from a state"""
        coords = np.copy(self.saved_coords)
        self.eigenvec = np.copy(self.saved_eigenvec)
        self.eigenval = self.saved_eigenval
        self.oldeigenvec = np.copy(self.eigenvec)
        self.overlap = self.saved_overlap
#        self.H0_leig = self.saved_H0_leig
        self.energy = self.saved_energy
        self.gradient = self.saved_gradient.copy()
        return coords

    def _compute_gradients(self, coords):
        """compute the energy and gradient at the current position and store them for later use"""
        self.nfev += 1
        self.energy, self.gradient = self.pot.getEnergyGradient(coords)

    def _set_energy_gradient(self, energy, gradient):
        self.energy = energy
        self.gradient = gradient.copy()

    def get_energy(self):
        """return the already computed energy at the current position"""
        return self.energy
    
    def get_gradient(self):
        """return the already computed gradient at the current position"""
        return self.gradient

    def run(self):
        """The main loop of the algorithm"""
        coords = np.copy(self.coords)
        res = Result() #  return object
        res.message = []
        
        # if starting with positive curvature, disable negative eigenvalue check
        # this will be reenabled as soon as the eigenvector becomes negative
        negative_before_check =  10

        self._compute_gradients(coords)
        for i in xrange(self.nsteps):
            
            # get the lowest eigenvalue and eigenvector
            self.overlap = self._getLowestEigenVector(coords, i)
            overlap = self.overlap
            #print self.eigenval
            
            if self.eigenval < 0:
                negative_before_check -= 1
            
            # check to make sure the eigenvector is ok
            if (i == 0 or self.eigenval <= 0 or not self.check_negative or 
                (negative_before_check > 0 and not self.demand_initial_negative_vec)):
                self._saveState(coords)
                self.reduce_step = 0
            else:
                self.npositive += 1
                if self.npositive > self.npositive_max:
                    logger.warning( "positive eigenvalue found too many times. ending %s", self.npositive)
                    res.message.append( "positive eigenvalue found too many times %d" % self.npositive )
                    break
                if self.verbosity > 2:
                    logger.info("the eigenvalue turned positive. %s %s", self.eigenval, "Resetting last good values and taking smaller steps")
                coords = self._resetState()
                self.reduce_step += 1
            
            # step uphill along the direction of the lowest eigenvector
            coords = self._stepUphill(coords)

            # minimize the coordinates in the space perpendicular to the lowest eigenvector
            tangent_ret = self._minimizeTangentSpace(coords, energy=self.get_energy(), gradient=self.get_gradient())
            coords = tangent_ret.coords
            tangentrms = tangent_ret.rms


            # check if we are done and print some stuff
#            self._compute_gradients(coords) # this is unnecessary
            E = self.get_energy()
            grad = self.get_gradient()
            rms = np.linalg.norm(grad) * self.rmsnorm
            gradpar = np.dot(grad, self.eigenvec) / np.linalg.norm(self.eigenvec)
            
            if self.iprint > 0:
                if (i+1) % self.iprint == 0:
                    ostring = "findTS: %3d E %9g rms %8g eigenvalue %9g rms perp %8g grad par %9g overlap %g" % (
                                    i, E, rms, self.eigenval, tangentrms, gradpar, overlap)
                    extra = "  Evec search: %d rms %g" % (self.leig_result.nfev, self.leig_result.rms)
                    extra += "  Tverse search: %d step %g" % (self.tangent_result.nfev, 
                                                                    self.tangent_move_step)
                    extra += "  Uphill step:%g" % (self.uphill_step_size,)
                    logger.info("%s %s", ostring, extra)
            
            if callable(self.event):
                self.event(energy=E, coords=coords, rms=rms, eigenval=self.eigenval, stepnum=i)
            if rms < self.tol:
                break
            if self.nfail >= self.nfail_max:
                logger.warning("stopping findTransitionState.  too many failures in eigenvector search %s", self.nfail)
                res.message.append( "too many failures in eigenvector search %d" % self.nfail )
                break

            if i == 0 and self.eigenval > 0.:
                if self.verbosity > 1:
                    logger.warning("initial eigenvalue is positive - increase NEB spring constant?")
                if self.demand_initial_negative_vec:
                    logger.warning("            aborting transition state search")
                    res.message.append( "initial eigenvalue is positive %f" % self.eigenval )
                    break

        # done.  do one last eigenvector search because coords may have changed
        self._getLowestEigenVector(coords, i)

        # print some data
        if self.verbosity > 0 or self.iprint > 0:
            logger.info("findTransitionState done: %s %s %s %s %s", i, E, rms, "eigenvalue", self.eigenval)
    
        success = True
        # check if results make sense
        if self.eigenval >= 0.:
            if self.verbosity > 2:
                logger.info( "warning: transition state is ending with positive eigenvalue %s", self.eigenval)
            success = False
        if rms > self.tol:
            if self.verbosity > 2:
                logger.info("warning: transition state search appears to have failed: rms %s", rms)
            success = False
        if i >= self.nsteps:
            res.message.append( "maximum iterations reached %d" % i )
            
        # update nfev with the number of calls from the transverse walker
        if self._transverse_walker is not None:
            twres = self._transverse_walker.get_result()
            self.nfev += twres.nfev 

        #return results
        res.coords = coords
        res.energy = E
        res.eigenval = self.eigenval
        res.eigenvec = self.eigenvec
        res.grad = grad
        res.rms = rms
        res.nsteps = i
        res.success = success
        res.nfev = self.nfev
        return res

    def _getLowestEigenVector(self, coords, i, gradient=None):
        """compute the lowest eigenvector at position coords
        
        Parameters
        ----------
        coords : the current position
        i : the iteration number
        gradient : the gradient at coords
        """
        if "nsteps" in self.lowestEigenvectorQuenchParams:
            niter = self.lowestEigenvectorQuenchParams["nsteps"]
        else:
            niter = 100
            if self.verbosity > 3:
                print "Using default of", niter, "steps for finding lowest eigenvalue"
        optimizer = FindLowestEigenVector(coords, self.pot,
#                                    H0=self.H0_leig, 
                            eigenvec0=self.eigenvec, 
                            orthogZeroEigs=self.orthogZeroEigs, 
                            gradient=gradient,
                            **self.lowestEigenvectorQuenchParams)
        res = optimizer.run(niter)
        if res.nsteps == 0:
            if self.verbosity > 2:
                print "eigenvector converged, but doing one iteration anyway"
            optimizer.one_iteration()
            res = optimizer.get_result()

        self.leig_result = res
        self.nfev += res.nfev
        
#        self.H0_leig = res.H0
        self.eigenvec = res.eigenvec
        self.eigenval = res.eigenval
        
        if i > 0:
            overlap = np.dot(self.oldeigenvec, res.eigenvec)
            if overlap < 0.5 and self.verbosity > 2:
                logger.info("warning: the new eigenvector has low overlap with previous %s %s", overlap, self.eigenval)
        else:
            overlap = 0.

        if res.success:
            self.nfail = 0
        else:
            self.nfail += 1
        
        self.oldeigenvec = self.eigenvec.copy()
        return overlap


    def _minimizeTangentSpace(self, coords, energy=None, gradient=None):
        """minimize the energy in the space perpendicular to eigenvec.
        
        Parameters
        ----------
        coords : the current position
        energy, gradient : the energy and gradient at the current position
        """
        assert gradient is not None
        if self._transverse_walker is None:
            if self.invert_gradient:
                # note: if we pass transverse energy and gradient here we can save 1 potential call
                self._transverse_walker = _DimerTranslator(coords, self.pot, self.eigenvec,
#                                         energy=transverse_energy, gradient=transverse_gradient,
                                         **self.tangent_space_quench_params)
            else:
                self._transverse_walker = _TransverseWalker(coords, self.pot, self.eigenvec, energy, gradient,
                                                            **self.tangent_space_quench_params)
        else:
            self._transverse_walker.update_eigenvec(self.eigenvec, self.eigenval)
            self._transverse_walker.update_coords(coords, energy, gradient)
        
        #determine the number of steps
        #i.e. if the eigenvector is deemed to have converged
        eigenvec_converged = self.overlap > .999 
        nstepsperp = self.nsteps_tangent1
        if eigenvec_converged:
            nstepsperp = self.nsteps_tangent2

        # reduce the maximum step size if necessary
        maxstep = self.maxstep_tangent
        if self.reduce_step > 0:
            maxstep *= (self.step_factor)**self.reduce_step
        self._transverse_walker.update_maxstep(maxstep)

        coords_old = coords.copy()
        ret = self._transverse_walker.run(nstepsperp)

        coords = ret.coords
        self.tangent_move_step = np.linalg.norm(coords - coords_old)
        self.tangent_result = ret
        if self.tangent_move_step > 1e-16:
            try:
                self.energy, self.gradient = self._transverse_walker.get_true_energy_gradient(coords)
            except AttributeError:
                print "was tspot was never called? use the same gradient"
                raise
        return ret

    def _update_max_uphill_step(self, Fold, stepsize):
        """use a trust radius to update the maximum uphill step size"""
        Fnew = np.dot(self.eigenvec, self.get_gradient())
        # EPER=MIN(DABS(1.0D0-(FOBNEW-FOB)/(PSTEP*EVALMIN)),DABS(1.0D0-(-FOBNEW-FOB)/(PSTEP*EVALMIN)))
        a1 = 1. - (Fnew - Fold) / (stepsize * self.eigenval)
        a2 = 1. - (-Fnew - Fold) / (stepsize * self.eigenval)
        eper = min(np.abs(a1), np.abs(a2))
        if eper > self._trust_radius:
            # reduce the maximum step size
            self._max_uphill = max(self._max_uphill / 1.1, self._max_uphill_min)
            if self.verbosity > 2:
                print "decreasing max uphill step to", self._max_uphill, "Fold", Fold, "Fnew", Fnew, "eper", eper, "eval", self.eigenval 
        else:
            # increase the maximum step size
            self._max_uphill = min(self._max_uphill * 1.1, self._max_uphill_max)
            if self.verbosity > 2:
                print "increasing max uphill step to", self._max_uphill, "Fold", Fold, "Fnew", Fnew, "eper", eper, "eval", self.eigenval


    def _stepUphill(self, coords):
        """step uphill in the direction of self.eigenvec.
        
        this is an eigenvector following step uphill.  The equation for the step size
        is described in 
        
        DJ Wales, J chem phys, 1994, 101, 3750--3762
        Rearrangements of 55-atom lennard-jones and (c-60)(55) clusters
        
        self.eigenval is used to determine the best stepsize
        """
        # the energy and gradient are already known
        e = self.get_energy()
        grad = self.get_gradient()
        F = np.dot(grad, self.eigenvec) 
        h = 2. * F / np.abs(self.eigenval) / (1. + np.sqrt(1. + 4. * (F / self.eigenval)**2))

        # get the maxstep and scale it if necessary
        maxstep = self._max_uphill
        if self.reduce_step > 0:
            maxstep *= (self.step_factor)**self.reduce_step


        if np.abs(h) > maxstep:
            if self.verbosity >= 5:
                logger.debug("reducing uphill step from %s %s %s", h, "to", maxstep) 
            h *= maxstep / np.abs(h)
        self.uphill_step_size = h
        coords += h * self.eigenvec

        # recompute the energy and gradient
        self._compute_gradients(coords)
        
        # update the maximum step using a trust ratio
        if self.eigenval < 0:
            self._update_max_uphill_step(F, h)

        if self.verbosity > 2:
            print "stepping uphill with stepsize", h

        return coords


def findTransitionState(*args, **kwargs):
    """
    simply a wrapper for initializing and running FindTransitionState
    
    See Also
    --------
    FindTransitionState : for all documentation
    """
    finder = FindTransitionState(*args, **kwargs)
    return finder.run()



        
        

###################################################################
#below here only stuff for testing
###################################################################


def testgetcoordsLJ():
    a = 1.12 #2.**(1./6.)
    theta = 60./360*np.pi
    coords = [ 0., 0., 0., \
              -a, 0., 0., \
              -a/2, a*np.cos(theta), 0., \
              -a/2, -a*np.cos(theta), 0.3 \
              ]
    coords = np.array(coords)
    return coords


def guesstsATLJ():
    from pele.potentials.ATLJ import ATLJ
    pot = ATLJ(Z = 2.)
    a = 1.12 #2.**(1./6.)
    theta = 60./360*np.pi
    coords1 = np.array([ 0., 0., 0., \
              -a, 0., 0., \
              -a/2, -a*np.cos(theta), 0. ])
    coords2 = np.array([ 0., 0., 0., \
              -a, 0., 0., \
              a, 0., 0. ])
    from pele.optimize import lbfgs_py as quench
    from pele.transition_states import InterpolatedPath
    ret1 = quench(coords1, pot.getEnergyGradient)
    ret2 = quench(coords2, pot.getEnergyGradient)
    coords1 = ret1[0]
    coords2 = ret2[0]
    from pele.transition_states import NEB
    neb = NEB(InterpolatedPath(coords1, coords2, 30), pot)
    neb.optimize()
    neb.MakeAllMaximaClimbing()
    #neb.optimize()
    for i in xrange(len(neb.energies)):
        if(neb.isclimbing[i]):
            coords = neb.coords[i,:]
    return pot, coords

def guessts(coords1, coords2, pot):
    from pele.optimize import lbfgs_py as quench
#    from pele.mindist.minpermdist_stochastic import minPermDistStochastic as mindist
    from pele.transition_states import NEB
    from pele.systems import LJCluster
    ret1 = quench(coords1, pot.getEnergyGradient)
    ret2 = quench(coords2, pot.getEnergyGradient)
    coords1 = ret1[0]
    coords2 = ret2[0]
    natoms = len(coords1)/3
    system = LJCluster(natoms)
    mindist = system.get_mindist()
    dist, coords1, coords2 = mindist(coords1, coords2) 
    print "dist", dist
    print "energy coords1", pot.getEnergy(coords1)
    print "energy coords2", pot.getEnergy(coords2)
    from pele.transition_states import InterpolatedPath
    neb = NEB(InterpolatedPath(coords1, coords2, 20), pot)
    #neb.optimize(quenchParams={"iprint" : 1})
    neb.optimize(iprint=-30, nsteps=100)
    neb.MakeAllMaximaClimbing()
    #neb.optimize(quenchParams={"iprint": 30, "nsteps":100})
    for i in xrange(len(neb.energies)):
        if(neb.isclimbing[i]):
            coords = neb.coords[i,:]
    return pot, coords, neb.coords[0,:], neb.coords[-1,:]


def guesstsLJ():
    from pele.potentials.lj import LJ
    pot = LJ()
    natoms = 9
    coords = np.random.uniform(-1,1,natoms*3)
    from pele.basinhopping import BasinHopping
    from pele.takestep.displace import RandomDisplacement
    from pele.takestep.adaptive import AdaptiveStepsize
    from pele.storage.savenlowest import SaveN
    saveit = SaveN(10)
    takestep1 = RandomDisplacement()
    takestep = AdaptiveStepsize(takestep1, frequency=15)
    bh = BasinHopping(coords, pot, takestep, storage=saveit, outstream=None)
    bh.run(100)
    coords1 = saveit.data[0].coords
    coords2 = saveit.data[1].coords
    
    return guessts(coords1, coords2, pot)

def testgetcoordsATLJ():
    a = 1.12 #2.**(1./6.)
    theta = 40./360*np.pi
    coords = [ 0., 0., 0., \
              -a, 0., 0., \
              a*np.cos(theta), a*np.sin(theta), 0. ]
    return np.array(coords)

def testpot1():
    import itertools
    pot, coords, coords1, coords2 = guesstsLJ()
    coordsinit = np.copy(coords)
    natoms = len(coords)/3
    c = np.reshape(coords, [-1,3])
    for i, j in itertools.combinations(range(natoms), 2):
        r = np.linalg.norm(c[i,:] - c[j,:])
        print i, j, r 
    
    e, g = pot.getEnergyGradient(coords)
    print "initial E", e
    print "initial G", g, np.linalg.norm(g)
    print ""
    

    
    
    from pele.utils.xyz import write_xyz

    #print ret
    
    with open("out.xyz", "w") as fout:
        e = pot.getEnergy(coords1)
        print "energy of minima 1", e
        write_xyz(fout, coords1, title=str(e))
        e, grad = pot.getEnergyGradient(coordsinit)
        print "energy of NEB guess for the transition state", e, "rms grad", \
            np.linalg.norm(grad) / np.sqrt(float(len(coords))/3.)
        write_xyz(fout, coordsinit, title=str(e))
        e = pot.getEnergy(coords2)
        print "energy of minima 2", e
        write_xyz(fout, coords2, title=str(e))
        
        #mess up coords a bit
        coords += np.random.uniform(-1,1,len(coords))*0.05
        e = pot.getEnergy(coords)
        write_xyz(fout, coords, title=str(e))

        
        printevent = PrintEvent(fout)
        print ""
        print "starting the transition state search"
        ret = findTransitionState(coords, pot, iprint=-1)
        print ret
        #coords, eval, evec, e, grad, rms = ret
        e = pot.getEnergy(ret.coords)
        write_xyz(fout, coords2, title=str(e))

    print "finished searching for transition state"
    print "energy", e
    print "rms grad", ret.rms
    print "eigenvalue", ret.eigenval
    
    if False:
        print "now try the same search with the dimer method"
        from pele.NEB.dimer import findTransitionState as dimerfindTS
        coords = coordsinit.copy()
        tau = np.random.uniform(-1,1,len(coords))
        tau /= np.linalg.norm(tau)
        ret = dimerfindTS(coords, pot, tau )
        enew, grad = pot.getEnergyGradient(ret.coords)
        print "energy", enew
        print "rms grad", np.linalg.norm(grad) / np.sqrt(float(len(ret.coords))/3.)



if __name__ == "__main__":
    testpot1()
