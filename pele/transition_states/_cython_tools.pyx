"""
note: there are two definitions of the spring force function because the (almost) pure
c version is so very ugly compared to the numpy version.  I have trouble believing that
this function, which is almost pure linear algebra is so much faster in pure c.

profiling and benchmarking needs to be done on this.
"""
import numpy as np
cimport numpy as np
cimport cython


def _neb_force(np.ndarray[double, ndim=1] t,
               np.ndarray[double, ndim=1] greal, 
               double d_left, 
               np.ndarray[double, ndim=1] g_left, 
               double d_right, 
               np.ndarray[double, ndim=1] g_right, 
               double k,
               dneb=True, with_springenergy=True):
    # project out parallel part
    gperp = greal - np.dot(greal, t) * t
    
    # parallel part of spring force
    gs_par = k*(d_left - d_right)*t
                
    g_tot = gperp + gs_par

    if dneb:
        g_spring = k*(g_left + g_right)
        # perpendicular part of spring
        gs_perp = g_spring - np.dot(g_spring,t)*t            
        # double nudging
        g_tot += gs_perp - np.dot(gs_perp,gperp)*gperp/np.dot(gperp,gperp)
    
    E = 0.
    if with_springenergy and dneb:
        E = 0.5 / k * np.dot(g_spring, g_spring)
    elif with_springenergy:
        # js850> I'm not sure if this is correct ... 
        E = 0.5 / k * (d_left **2 + d_right**2)
    #print np.linalg.norm(gperp), np.linalg.norm(gs_par)
    return E, g_tot

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def _neb_force_c_like(np.ndarray[double, ndim=1] t,
               np.ndarray[double, ndim=1] greal, 
               double d_left, 
               np.ndarray[double, ndim=1] g_left, 
               double d_right, 
               np.ndarray[double, ndim=1] g_right, 
               double k,
               dneb=True, with_springenergy=True):
    cdef int i, n
    n = t.size
    cdef np.ndarray[double, ndim=1] gperp = np.zeros(n)
    cdef np.ndarray[double, ndim=1] g_tot = np.zeros(n)
    cdef np.ndarray[double, ndim=1] g_spring = np.zeros(n)
    cdef np.ndarray[double, ndim=1] gs_perp = np.zeros(n)
#     gperp = np.zeros(n)
#     g_tot = np.zeros(n)
#     g_spring = np.zeros(n)
#     gs_perp = np.zeros(n)

    # project out parallel part
    cdef double greal_dot_t = 0
    for i in xrange(n):
        greal_dot_t += greal[i] * t[i]
    for i in xrange(n):
        gperp[i] = greal[i] - greal_dot_t * t[i]
    
    # parallel part of spring force
#     cdef np.ndarray[double, ndim=1] gs_par = np.zeros()
    
#     gs_par = k*(d_left - d_right)*t
#     g_tot = gperp + gs_par
    for i in xrange(n):
        g_tot[i] = gperp[i] + k * (d_left - d_right) * t[i]

    cdef double g_spring_dot_t = 0
    cdef double gs_perp_dot_gperp = 0.
    cdef double gperp_dot_gperp = 0
    cdef double gspring_dot_gspring = 0
    if dneb:
        for i in xrange(n):
            g_spring[i] = k * (g_left[i] + g_right[i])
            
        # perpendicular part of spring
        for i in xrange(n):
            g_spring_dot_t += g_spring[i] * t[i]
        for i in xrange(n):
            gs_perp[i] = g_spring[i] - g_spring_dot_t * t[i]
                        
        # double nudging
        for i in xrange(n):
            gs_perp_dot_gperp += gs_perp[i] * gperp[i]
        for i in xrange(n):
            gperp_dot_gperp += gperp[i] * gperp[i]
        for i in xrange(n):
            g_tot[i] += gs_perp[i] - gs_perp_dot_gperp * gperp[i] / gperp_dot_gperp
    
    E = 0.
    if with_springenergy and dneb:
        gspring_dot_gspring = 0
        for i in xrange(n):
            gspring_dot_gspring += g_spring[i] * g_spring[i]
        E = 0.5 / k * gspring_dot_gspring
    elif with_springenergy:
        # js850> I'm not sure if this is correct ... 
        E = 0.5 / k * (d_left **2 + d_right**2)
    #print np.linalg.norm(gperp), np.linalg.norm(gs_par)
    return E, g_tot
