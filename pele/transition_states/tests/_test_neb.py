import unittest

import numpy as np
# from pele.transition_states._NEB_utils import neb_force
from pele.transition_states._cython_tools import _neb_force

def _pythonic_neb_force(t, greal, d_left, g_left, d_right, g_right, k,
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

class TestNebForce(unittest.TestCase):
    def check(self, dneb=True):
        n = 3*4
        greal1 = np.random.uniform(-1,1,n)
        g_left = np.random.uniform(-1,1,n)
        d_right = float(np.random.uniform(-1,1))
        d_left = float(np.random.uniform(-1,1))
        g_right = np.random.uniform(-1,1,n)
        k = 10.
        t = np.random.uniform(-1,1,n)
        print greal1.shape
        greal1 = np.array(greal1, order='F')
        e, g_tot = _pythonic_neb_force(t, greal1, d_left, g_left, d_right, g_right, k, dneb=dneb)
        e1, g_tot2 = _neb_force(t, greal1, d_left, g_left, d_right, g_right, k, dneb)
        self.assertAlmostEqual(e, e1, 3)
        for g1, g2 in zip(g_tot, g_tot2):
            self.assertAlmostEqual(g1, g2, 3)
        
    def test_with_dneb(self):
        self.check(dneb=True)
    def test_no_dneb(self):
        self.check(dneb=False)

if __name__ == "__main__":
    unittest.main()
