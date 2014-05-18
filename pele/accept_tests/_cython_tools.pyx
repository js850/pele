cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def _check_spherical_container(x,
                               double radius2,
                               center=False):
    x = np.asarray(x, float).reshape(-1,3)
    cdef np.ndarray[double, ndim=2] coords = np.asarray(x, float)
    #get center of mass
    cdef int natoms = coords.shape[0]
    cdef np.ndarray[double, ndim=1] com
    if center:
        com = np.sum(coords, 0) / natoms
    else:
        com = np.zeros(3)

    cdef int i, k
    cdef int n = coords.size
    cdef double r, r2
    cdef int in_sphere = True
    for i in xrange(natoms):
        r2 = 0
        for k in xrange(3):
            r = coords[i,k] - com[k]
            r2 += r*r
        if r2 > radius2:
            in_sphere = False
            break
    
    return bool(in_sphere)

                               
                           
