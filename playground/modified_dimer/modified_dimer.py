from itertools import izip

import numpy as np
from matplotlib import pyplot as plt

from pele.potentials import BasePotential
from pele.utils.rotations import vec_random_ndim

class Position(object):
    x = None
    v = None
    lam = None
    e = None
    g = None
    e_dv = None
    g_dv = None
    e_dl = None
    g_dl = None
    e_dv_dl = None
    g_dv_dl = None
    
    def __init__(self, potential, x, v, lam, dx=1e-7):
        self.potential = potential
        self.x = x
        self.v = v
        self.lam = lam
        self.dx = dx
    
    def compute_energies(self):
        x = self.x
        v = self.v
        lam = self.lam / np.linalg.norm(self.lam)
        self.e, self.g = self.potential.getEnergyGradient(x)
        self.e_dv, self.g_dv = self.potential.getEnergyGradient(x + self.dx * v)
        self.e_dl, self.g_dl = self.potential.getEnergyGradient(x + self.dx * lam)
        self.e_dv_dl, self.g_dv_dl = self.potential.getEnergyGradient(x + self.dx * lam + self.dx * v)
        self.hess = self.potential.getHessian(x)
        self.hess_dv = self.potential.getHessian(x + self.dx * v)

class CurvaturePotential(BasePotential):
    def __init__(self, potential, dx=1e-7):
        self.potential = potential
        self.dx = float(dx)
        self.with_hess = True
    

    def curvature_along_v(self, p):
        if self.with_hess:
            return np.dot(p.v, np.dot(p.hess, p.v))
        else:
            return np.dot(p.v, (p.g_dv - p.g)) / p.dx
    def curvature_along_lam(self, p):
        return np.dot(p.lam, (p.g_dl - p.g)) / p.dx
    def grad_curvature_v_mu_dot_lam(self, p):
        return (p.g_dv_dl - p.g_dv - p.g_dl - p.g) / p.dx**2
    def grad_curvature_wrt_v(self, p):
        if self.with_hess:
            v_hess = np.dot(p.v, p.hess)
            mu = np.dot(p.v, v_hess)
            return 2.* v_hess - 2. * mu * p.v 
        else:
            dg = (p.g_dv - p.g) / p.dx
            mu = np.dot(dg, p.v)
            
            dmu_dv = 2 * dg - mu * p.v
            return dmu_dv
    
    def update_coords(self, x, v, lam):
        v /= np.linalg.norm(v)
        self.pos = Position(self.potential, x, v, lam, dx=self.dx)
        self.pos.compute_energies()
        return self.pos
    
    def test(self, x, v, lam):
        pos = Position(self.potential, x, v, lam, dx=self.dx)
        pos.compute_energies()
        self.pos = pos
        
        hess = self.potential.getHessian(x)
        mu_v = self.curvature_along_v(pos)
        print "the curvature along the v direction is", mu_v, " it should be", np.dot(v, np.dot(hess, v))
        
        mu_lam = self.curvature_along_lam(pos)
        print "the curvature along the lam direction is", mu_lam, " it should be", np.dot(lam, np.dot(hess, lam)) / np.dot(lam,lam)

        dmu_v_lam = self.grad_curvature_v_mu_dot_lam(pos)
        print "the change in curvature along the v direction w.r.t. changes in the lam direction", dmu_v_lam

    def split_x_v_lam(self, x_v_lam):
        x_v_lam = x_v_lam.reshape([3,-1])
        return x_v_lam
    def merge_x_v_lam(self, x, v, lam):
        x_v_lam = np.zeros([3, x.size])
        x_v_lam[0,:] = x
        x_v_lam[1,:] = v
        x_v_lam[2,:] = lam
        return x_v_lam.ravel()

    def grad_wrt_lam(self, pos):
        return self.grad_curvature_wrt_v(pos)

    def grad_wrt_x(self, p):
        if self.with_hess:
            t1 = 2. * (np.dot(p.hess_dv, p.lam) - np.dot(p.hess, p.lam)) / p.dx
        else:
            t1 = 2. * (p.g_dv_dl - p.g_dv - p.g_dl + p.g) / p.dx**2 * np.linalg.norm(p.lam)
        if self.with_hess:
            dmu_dx = (np.dot(p.hess_dv, p.v) - np.dot(p.hess, p.v)) / p.dx
            t2 = -2. * np.dot(p.v, p.lam) * dmu_dx
        else:
            raise NotImplementedError
        print "grad wrt x"
        print t1
        print t2
        return t1 + t2

    def grad_wrt_v(self, p):
        dmu_dv = self.grad_curvature_wrt_v(p)
        mu = self.curvature_along_v(p)
        t1 = -2. * np.dot(p.v, p.lam) * dmu_dv
        if self.with_hess:
            t2 = 2. * np.dot(p.lam, p.hess)
        else:
            t2 = 2. * np.linalg.norm(p.lam) * (p.g_dl - p.g) / p.dx
            
        t3 = -2. * mu * p.lam
        t4 = -2. * np.dot(p.lam, dmu_dv) * p.v
        
        print "grad wrt v"
        print t1
        print t2
        print t3
        print t4
        
        return t1 + t2 + t3 + t4
        
    def getEnergy(self, x_v_lam):
        x, v, lam = self.split_x_v_lam(x_v_lam)
        pos = self.update_coords(x, v, lam)
        
        dmu_dv = self.grad_curvature_wrt_v(pos)
        return np.dot(dmu_dv, lam)

    def getEnergyGradient1(self, x_v_lam):
        x, v, lam = self.split_x_v_lam(x_v_lam)
        pos = self.update_coords(x, v, lam)
        e = np.dot(lam, self.grad_curvature_wrt_v(pos))
        
        grad = self.merge_x_v_lam(self.grad_wrt_x(pos),
                                  self.grad_wrt_v(pos),
                                  self.grad_wrt_lam(pos))
        return e, grad


class ModifiedDimer(BasePotential):
    def __init__(self, potential):
        self.cpot = CurvaturePotential(potential)
    
    def getEnergy(self, x_v_lam):
        x, v, lam = self.cpot.split_x_v_lam(x_v_lam)
        return 0.
    
    def getEnergyGradient(self, x_v_lam):
        x, v, lam = self.cpot.split_x_v_lam(x_v_lam)
        v /= np.linalg.norm(v)
        e, gxvlam = self.cpot.getEnergyGradient(x_v_lam)
        true_grad = self.cpot.pos.g
        projected_grad = true_grad - 2. * np.dot(true_grad, v) * v
        g0 = np.zeros(projected_grad.size)
        g1 = self.cpot.merge_x_v_lam(projected_grad, g0, g0)
        return 0., g1 + gxvlam
        

class TestPot(BasePotential):
    def getEnergy(self, x):
        dx = np.zeros(x.size)
        dx[0] += 1
        return np.linalg.norm(x-dx)**2 * np.linalg.norm(x+dx)**2
    def getEnergyGradient(self, x):
        dx = np.zeros(x.size)
        dx[0] += 1
        e = self.getEnergy(x)
        return e, 2. * (x-dx) * np.linalg.norm(x+dx)**2 + 2. * np.linalg.norm(x-dx)**2 * (x+dx)
    
    def getEnergyGradientHessian(self, x):
        dx = np.zeros(x.size)
        dx[0] += 1
        e, g = self.getEnergyGradient(x)
        h = ( 2. * np.linalg.norm(x+dx)**2 * np.eye(2)
             +2. * np.linalg.norm(x-dx)**2 * np.eye(2)
             +4. * np.outer(x-dx, x+dx)
             +4. * np.outer(x+dx, x-dx))
        return e, g, h

def plot_test_pot_background():
    x0 = np.arange(-1.5, 1.5,.05)
    y0 = np.arange(-.6, .6,.05)
    xgrid, ygrid = np.meshgrid(x0, y0)
    p = TestPot()
    f = np.array([p.getEnergy(np.array([x1,y])) for x1, y in izip(xgrid.ravel(), ygrid.ravel())]).reshape(xgrid.shape)
    plt.contourf(x0, y0, f)

def plot_test_pot(x, v, lam):
    plot_test_pot_background()
    
    ax = plt.gca()
    h = x+v
    ax.arrow(x[0], x[1], h[0], h[1])
    h = x+lam
    ax.arrow(x[0], x[1], h[0], h[1], fc='r', ec='r')
    plt.show()

def test():
    p = TestPot()

    x = np.array([0.,.1])
    v = vec_random_ndim(2) 
    lam = vec_random_ndim(2)
#    v = np.array([.4,.1])
#    v /= np.linalg.norm(v) 
#    lam = np.array([.3,-.1])

    cpot = CurvaturePotential(p)
    cpot.test(x, v, lam)
    
    xvlam = cpot.merge_x_v_lam(x, v, lam)
    e = cpot.getEnergy(xvlam)
    print e
    
    e, g = cpot.getEnergyGradient(xvlam)
    print e
    print g
    
    ng = cpot.NumericalDerivative(xvlam, eps=1e-4)
    print "gradient", g
    print "num grad", ng
    
#    eps = 1e-6
#    x = xvlam.copy()
#    x[0] += eps
#    ep = cpot.getEnergy(x)
#    x = xvlam.copy()
#    x[0] -= eps
#    em = cpot.getEnergy(x)
#    print ep, em, (ep - em) / 2./eps
    
#    f = np.array([cpot.getEnergy(cpot.merge_x_v_lam(np.array([x,y]), v, lam)) 
#                  for x, y in izip(xgrid.ravel(), ygrid.ravel())]).reshape(xgrid.shape)
#    plt.contourf(x0, y0, f)
#    plt.colorbar()
#    plt.show()

    plot_test_pot(x, v, lam)

def draw_arrow(x, v, ax, c='k'):
    h = x+v
    ax.arrow(x[0], x[1], h[0], h[1], fc=c, ec=c)

def test2():
    from pele.optimize._quench import steepest_descent
    from pele.utils.hessian import get_smallest_eig
    p = TestPot()
    x = np.array([0.1, 0.1])
    v = vec_random_ndim(2) 
    lam = vec_random_ndim(2)
    lam /= 100.
#    v = np.array([.4,.1])
#    v /= np.linalg.norm(v) 
#    lam = np.array([.3,-.1])

    hess = p.getHessian(x)
    mu, v = get_smallest_eig(hess)

#    plot_test_pot(x, v, lam)

    cpot = CurvaturePotential(p)
    dpot = ModifiedDimer(p)
    
    xvlam = cpot.merge_x_v_lam(x, v, lam)
    
    xvl_list = []
    def callback(coords=None, **kwargs):
        xvl_list.append(coords.copy())
    
    
    steepest_descent(xvlam, dpot, iprint=1, dx=1e-2, nsteps=50, events=[callback])
    
    plot_test_pot_background()
    ax = plt.gca()
    for xvl in xvl_list:
        x, v, l = cpot.split_x_v_lam(xvl)
        print x, v, lam
#        plt.scatter(x[0], x[1])
        draw_arrow(x, v, ax)
#        draw_arrow(x, lam, ax, c='r')
    
    plt.show()
    
    

    
if __name__ == "__main__":
    test2()