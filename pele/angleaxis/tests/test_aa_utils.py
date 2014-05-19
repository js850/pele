import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pele.angleaxis._aadist import rmdrvt, sitedist_grad
from pele.utils.rotations import vec_random

def _rot_mat_derivative_small_theta(p, with_grad):
    theta2 = p.dot(p)
    if theta2 > 1e-2:
        raise ValueError("theta must be small")
    # Execute if the angle of rotation is zero
    # In this case the rotation matrix is the identity matrix
    rm = np.eye(3)

    # vr274> first order corrections to rotation matrix
    rm[1,0] = -p[2]
    rm[0,1] = p[2]
    rm[2,0] = p[1]
    rm[0,2] = -p[1]
    rm[2,1] = -p[0]
    rm[1,2] = p[0]

    #        If derivatives do not need to found, we're finished
    if not with_grad: 
        return rm, None, None, None 

    # hk286 - now up to the linear order in theta
    e = np.zeros([3,3])
    e[0,0]    = 0.0
    e[1,0]    = p[1]
    e[2,0]    = p[2]
    e[0,1]    = p[1]
    e[1,1]    = -2.0*p[0]
    e[2,1]    = -2.0
    e[0,2]    = p[2]
    e[1,2]    = 2.0
    e[2,2]    = -2.0*p[0]
    drm1 = 0.5*e

    e[:]    = 0.
    e[0,0]    = -2.0*p[1]
    e[1,0]    = p[0]
    e[2,0]    = 2.0
    e[0,1]    = p[0]
    e[1,1]    = 0.0
    e[2,1]    = p[2]
    e[0,2]    = -2.0
    e[1,2]    = p[2]
    e[2,2]    = -2.0*p[1]
    drm2 = 0.5*e

    e[:] = 0
    e[0,0]    = -2.0*p[2]
    e[1,0]    = -2.0
    e[2,0]    = p[0]
    e[0,1]    = 2.0
    e[1,1]    = -2.0*p[2]
    e[2,1]    = p[1]
    e[0,2]    = p[0]
    e[1,2]    = p[1]
    e[2,2]    = 0.0
    drm3 = 0.5*e
    
    return rm, drm1, drm2, drm3
    

def _rot_mat_derivative(p, with_grad):
    """compute the derivative of a rotation matrix
    
    Parameters
    ----------
    p : ndarray
        angle axis vector
    """
    theta2 = p.dot(p)
    if theta2 < 1e-12: return _rot_mat_derivative_small_theta(p, with_grad)
    # Execute for the general case, where THETA dos not equal zero
    # Find values of THETA, CT, ST and THETA3
    theta   = np.sqrt(theta2)
    ct      = np.cos(theta)
    st      = np.sin(theta)
    theta3  = 1./(theta2*theta)

    # Set THETA to 1/THETA purely for convenience
    theta   = 1./theta

    # Normalise p and construct the skew-symmetric matrix E
    # ESQ is calculated as the square of E
    pn   = theta * p
    e = np.zeros([3,3])
    e[0,1]  = -pn[2]
    e[0,2]  =  pn[1]
    e[1,2]  = -pn[0]
    e[1,0]  = -e[0,1]
    e[2,0]  = -e[0,2]
    e[2,1]  = -e[1,2]
    esq     = np.dot(e, e)

    # RM is calculated from Rodrigues' rotation formula [equation [1]
    # in the paper]
    rm      = np.eye(3) + [1. - ct] * esq + st * e
    
    if not with_grad:
        return rm, None, None, None

    # If derivatives do not need to found, we're finished

    # Set up DEk using the form given in equation (4) in the paper
    de1 = np.zeros([3,3])
    de1[0,1] = p[2]*p[0]*theta3
    de1[0,2] = -p[1]*p[0]*theta3
    de1[1,2] = -(theta - p[0]*p[0]*theta3)
    de1[1,0] = -de1[0,1]
    de1[2,0] = -de1[0,2]
    de1[2,1] = -de1[1,2]

    de2 = np.zeros([3,3])
    de2[0,1] = p[2]*p[1]*theta3
    de2[0,2] = theta - p[1]*p[1]*theta3
    de2[1,2] = p[0]*p[1]*theta3
    de2[1,0] = -de2[0,1]
    de2[2,0] = -de2[0,2]
    de2[2,1] = -de2[1,2]

    de3 = np.zeros([3,3])
    de3[0,1] = -(theta - p[2]*p[2]*theta3)
    de3[0,2] = -p[1]*p[2]*theta3
    de3[1,2] = p[0]*p[2]*theta3
    de3[1,0] = -de3[0,1]
    de3[2,0] = -de3[0,2]
    de3[2,1] = -de3[1,2]

    # Use equation (3) in the paper to find DRMk
    drm1 = (st*pn[0]*esq + (1.-ct)*(de1.dot(e) + e.dot(de1))
            + ct*pn[0]*e + st*de1)

    drm2 = (st*pn[1]*esq + (1.-ct)*(de2.dot(e) + e.dot(de2))
            + ct*pn[1]*e + st*de2)

    drm3 = (st*pn[2]*esq + (1.-ct)*(de3.dot(e) + e.dot(de3))
            + ct*pn[2]*e + st*de3)
    
    return rm, drm1, drm2, drm3

def _sitedist_grad(drij, p1, p2, S, W, cog):
    """
    Parameters
    ----------
    drij : length 3 array
        shortest vector from com1 to com2
    p1, p2 : length 3 array
        angle axis vectors for the two rigid bodies
    S : 3x3 array
        weighted tensor of gyration S_ij = \sum m_i x_i x_j 
    W : float
        sum of all weights
    cog : 3 dim np.array
        center of gravity
    """
    R2 = _rot_mat_derivative(p2, False)[0]
    R1, R11, R12, R13 = _rot_mat_derivative(p1, True)

    dR = R2 - R1

    g_M = -2. * W * drij

    g_P = np.zeros(3)
    g_P[0] = -2. * np.trace(np.dot(R11, np.dot(S, np.transpose(dR))))
    g_P[1] = -2. * np.trace(np.dot(R12, np.dot(S, np.transpose(dR))))
    g_P[2] = -2. * np.trace(np.dot(R13, np.dot(S, np.transpose(dR))))

    g_M -= 2. * W *  np.dot(dR, cog)
    g_P[0] -= 2. * W * np.dot(drij, np.dot(R11, cog))
    g_P[1] -= 2. * W * np.dot(drij, np.dot(R12, cog))
    g_P[2] -= 2. * W * np.dot(drij, np.dot(R13, cog))

    return g_M, g_P

class TestRmDrvt(unittest.TestCase):
    def test1(self):
        P = vec_random() * np.random.uniform(1e-5,1)
        self.check(P, True)

    def test_small_theta(self):
        P = vec_random() * np.random.uniform(0,1e-7)
        print P
        self.check(P, True)
    
    def check(self, P, with_grad):
        rm, drm1, drm2, drm3 = rmdrvt(P, with_grad)
        rmp, drm1p, drm2p, drm3p = _rot_mat_derivative(P, with_grad)
        print rm
        print rmp
        assert_array_almost_equal(rm, rmp, decimal=4)
        assert_array_almost_equal(drm1, drm1p, decimal=4)
        assert_array_almost_equal(drm2, drm2p, decimal=4)
        assert_array_almost_equal(drm3, drm3p, decimal=4)
    
    def test_big_small(self):
        P = vec_random()
        P1 = P * (1.1e-6)
        P2 = P * (0.9e-6)
        print P1.dot(P1) > 1e-12, P2.dot(P2) > 1e-12

        rm, drm1, drm2, drm3 = rmdrvt(P1, True)
        rmp, drm1p, drm2p, drm3p = rmdrvt(P1, True)
#        print rm
#        print rmp
        assert_array_almost_equal(rm, rmp, decimal=4)
        assert_array_almost_equal(drm1, drm1p, decimal=4)
        assert_array_almost_equal(drm2, drm2p, decimal=4)
        assert_array_almost_equal(drm3, drm3p, decimal=4)

class TestSiteDistGrad(unittest.TestCase):
    def test1(self):
        from pele.utils.rotations import vec_random
        drij = np.random.uniform(-1,1,3)
        p1 = vec_random() * 0.5
        p2 = vec_random() * 0.5
        W = 1.3
        S = np.random.uniform(-1,1,[3,3])
        cog = np.random.uniform(-1,1,3)
        
        g_M, g_P = sitedist_grad(drij, p1, p2, S, W, cog)
        g_Mp, g_Pp = _sitedist_grad(drij, p1, p2, S, W, cog)
        assert_array_almost_equal(g_M, g_Mp, decimal=4)
        assert_array_almost_equal(g_P, g_Pp, decimal=4)
    
if __name__ == "__main__":
    unittest.main()
