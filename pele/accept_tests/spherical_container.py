import numpy as np
import pele.exceptions as exc
from pele.accept_tests._cython_tools import _check_spherical_container

__all__ = ["SphericalContainer"]

class SphericalContainer(object):
    """
    Reject a structure if any atoms are outside a spherical region

    This test is necessary to avoid evaporation in clusters
    
    a class to make sure the cluster doesn't leave a spherical region of
    given radius.  The center of the spherical region is at the center of mass.
    
    Parameters
    ----------
    radius : float
    """
    def __init__(self, radius, nocenter=False, verbose=False):
        if radius < 0:
            raise exc.SignError
        self.radius = float(radius)
        self.radius2 = float(radius)**2
        self.count = 0
        self.nrejected = 0
        self.nocenter = nocenter
        self.verbose = verbose
        
    
    def accept(self, coords):
        """ perform the test"""
        self.count += 1
        in_sphere = _check_spherical_container(coords, self.radius2, not self.nocenter)
        #get center of mass
        if not in_sphere and self.verbose: 
            self.nrejected += 1
            print "radius> rejecting", self.nrejected, "out of", self.count
        return in_sphere
    
    def __call__(self, enew, coordsnew, **kwargs):
        """wrapper for accept"""
        return self.accept(coordsnew)
