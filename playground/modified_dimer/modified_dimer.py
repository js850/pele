from itertools import izip

import numpy as np
from matplotlib import pyplot as plt

from pele.potentials import BasePotential
from pele.utils.rotations import vec_random_ndim
from pele.utils.hessian import get_smallest_eig

class Position(object):
    x = None
    v = None
    e = None
    g = None
    e_dv = None
    g_dv = None
    
    def __init__(self, potential, x, v, dx=1e-7):
        self.potential = potential
        self.x = x
        self.v = v
        self.dx = dx
        self.vnorm = np.linalg.norm(v)
    
    def compute_energies(self):
        x = self.x
        v = self.v
        self.e, self.g = self.potential.getEnergyGradient(x)
        self.e_dv, self.g_dv = self.potential.getEnergyGradient(x + self.dx * v)
        self.hess = self.potential.getHessian(x)
        self.hess_dv = self.potential.getHessian(x + self.dx * v)

class CurvaturePotential(BasePotential):
    def __init__(self, potential, dx=1e-7, lam=1.):
        self.potential = potential
        self.dx = float(dx)
        self.with_hess = True
        self.lam = lam
    

    def curvature_along_v(self, p):
        if self.with_hess:
            return np.dot(p.v, np.dot(p.hess, p.v)) / p.vnorm**2
        else:
            return np.dot(p.v, (p.g_dv - p.g)) / p.dx / p.vnorm**2
        
    def grad_curvature_wrt_v(self, p):
        if self.with_hess:
            v_hess = np.dot(p.v, p.hess)
            mu = np.dot(p.v, v_hess) / p.vnorm**2
            return (2.* v_hess - 2. * mu * p.v) / p.vnorm**2 
        else:
            dg = (p.g_dv - p.g) / p.dx
            mu = np.dot(dg, p.v) / p.vnorm**2
            
            dmu_dv = 2 * (dg - mu * p.v) / p.vnorm**2
            return dmu_dv
    
    def update_coords(self, x, v):
#        v /= np.linalg.norm(v)
        self.pos = Position(self.potential, x, v, dx=self.dx)
        self.pos.compute_energies()
        return self.pos
    
    def split_x_v(self, x_v):
        x_v = x_v.reshape([2,-1])
        return x_v
    def merge_x_v(self, x, v):
        x_v_lam = np.zeros([2, x.size])
        x_v_lam[0,:] = x
        x_v_lam[1,:] = v
        return x_v_lam.ravel()

    def grad_wrt_x(self, p):
        dmu_dv = self.grad_curvature_wrt_v(p)
        if self.with_hess:
            t1 = (np.dot(p.hess_dv, dmu_dv) - np.dot(p.hess, dmu_dv)) / p.dx
        else:
            raise NotImplementedError
        return 4. * self.lam * t1 / p.vnorm

    def grad_wrt_v(self, p):
        dmu_dv = self.grad_curvature_wrt_v(p)
        mu = self.curvature_along_v(p)
        if self.with_hess:
            t2 = np.dot(dmu_dv, p.hess)
        else:
            raise NotImplementedError
            
        t3 = - mu * dmu_dv
        t4 = - np.dot(dmu_dv, dmu_dv) * p.v
        
        return 4. * self.lam * (t2 + t3 + t4) / p.vnorm
        
    def getEnergy(self, x_v):
        x, v = self.split_x_v(x_v)
        pos = self.update_coords(x, v)
        
        dmu_dv = self.grad_curvature_wrt_v(pos)
        return self.lam * np.dot(dmu_dv, dmu_dv)

    def getEnergyGradient(self, x_v):
        x, v = self.split_x_v(x_v)
        pos = self.update_coords(x, v)

        dmu_dv = self.grad_curvature_wrt_v(pos)
        e = self.lam * np.dot(dmu_dv, dmu_dv)
        
        grad = self.merge_x_v(self.grad_wrt_x(pos),
                              self.grad_wrt_v(pos))
        grad[:x.size] = 0. # trial
#        grad[:] = 0. # debug
        return e, grad


class ModifiedDimer(BasePotential):
    def __init__(self, potential, **kwargs):
        self.cpot = CurvaturePotential(potential, **kwargs)
    
    def getEnergy(self, x_v):
        e = self.cpot.getEnergy(x_v)
        return e
    
    def getEnergyGradient(self, x_v):
        x, v = self.cpot.split_x_v(x_v)
#        v /= np.linalg.norm(v)
        e, gxv = self.cpot.getEnergyGradient(x_v)
        true_grad = self.cpot.pos.g
        projected_grad = true_grad - 2. * np.dot(true_grad, v) / np.dot(v,v) * v
#        projected_grad = -np.dot(true_grad, v) * v / np.dot(v,v) # debug
        g0 = np.zeros(projected_grad.size)
        g1 = self.cpot.merge_x_v(projected_grad, g0)
        gtot = g1 + gxv
        self.projected_grad = projected_grad.copy()
        self.curvature_term_grad = gxv.copy() / self.cpot.lam
        self.curvature_term_energy = e / self.cpot.lam
#        assert np.dot(gtot[:x.size], true_grad) <= 0
#        print np.dot(gtot[:x.size], true_grad)
#        print np.dot(gtot[:x.size], true_grad) / np.linalg.norm(gtot[:x.size]) / np.linalg.norm(true_grad)
        return e, gtot.copy()
        

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

class TestPot2(BasePotential):
    x1 = np.array([1.,0])
    x2 = np.array([-.8,0])
    L1 = .9
    L2 = .4
    def getEnergy(self, x):
        t1 = -3. * np.exp(-np.dot(x-self.x1, x-self.x1) / self.L1**2)
        t2 = -3.5 * np.exp(-np.dot(x-self.x2, x-self.x2) / self.L2**2)
        t3 = 0.01 * np.dot(x,x)
        t3 = 0.
        return t1 + t2 + t3
    def getEnergyGradient(self, x):
        t1 = -3. * np.exp(-np.dot(x-self.x1, x-self.x1) / self.L1**2)
        g1 = -2.*t1 * (x-self.x1) / self.L1**2 
        t2 = -3.5 * np.exp(-np.dot(x-self.x2, x-self.x2) / self.L2**2)
        g2 = -2.*t2 * (x-self.x2) / self.L2**2 
        t3 = 0.01 * np.dot(x,x)
        g3 = 2*0.01 * x
        t3 = g3 = 0;
        return t1 + t2 + t3, g1 + g2 + g3

def plot_test_pot_background(p=None, v=None, xmax=1.5, ymax=.6):
    x0 = np.linspace(-xmax, xmax, 100)
    y0 = np.linspace(-ymax, ymax, 100)
    xgrid, ygrid = np.meshgrid(x0, y0)
    p = p or TestPot()
    if v is None:
        f = np.array([p.getEnergy(np.array([x1,y])) for x1, y in izip(xgrid.ravel(), ygrid.ravel())]).reshape(xgrid.shape)
    else:
        f = np.array([p.getEnergy(p.cpot.merge_x_v(np.array([x1,y]),v)) for x1, y in izip(xgrid.ravel(), ygrid.ravel())]).reshape(xgrid.shape)
    plt.contourf(x0, y0, f)
    plt.colorbar()
    return x0, y0, xgrid, ygrid

def plot_test_pot(x, v):
    plot_test_pot_background()
    
    ax = plt.gca()
    h = x+v
    ax.arrow(x[0], x[1], h[0], h[1])
    plt.show()

class TestCPot(BasePotential):
    def __init__(self, cpot, x):
        self.cpot = cpot
        self.x = x.copy()
    
    def getEnergy(self, v):
        v = v.copy()
#        v = v / np.linalg.norm(v)
        pos = self.cpot.update_coords(self.x.copy(), v)
        return self.cpot.curvature_along_v(pos)
    def getEnergyGradient(self, v):
        v = v.copy()
#        v = v / np.linalg.norm(v)
        pos = self.cpot.update_coords(self.x.copy(), v.copy())
        mu = self.cpot.curvature_along_v(pos)
        return mu, self.cpot.grad_curvature_wrt_v(pos)
    def getEnergyGradientHessian(self, v):
        v = v.copy()
#        v = v / np.linalg.norm(v)
        mu, dmu_dv = self.getEnergyGradient(v)
        hess = 2. * (-np.outer(v, dmu_dv)
                     +self.cpot.pos.hess
                     -np.outer(dmu_dv, v)
                     -mu * np.eye(v.size)
                     ) / self.cpot.pos.vnorm**2
        return mu, dmu_dv, hess

def test():
    p = TestPot()

    x = np.array([0.,.1])
    v = vec_random_ndim(2)
    v = np.array([.4,.1])
    v /= np.linalg.norm(v) 
#    v = np.array([np.sqrt(2)]*2)
#    v = np.array([1.,0])

    cpot = CurvaturePotential(p)
    
    xv = cpot.merge_x_v(x, v)
    e = cpot.getEnergy(xv)
    print e
    
    e, g = cpot.getEnergyGradient(xv.copy())
    print "energy", e
    print g
    print "curvature", cpot.curvature_along_v(cpot.pos)
    
    ng = cpot.NumericalDerivative(xv, eps=1e-3)
    print "gradient", g
    print "num grad", ng
    
    
def test3():
    x = np.array([0.1,.2])
    v = vec_random_ndim(2)
    v = np.array([.4,.1])
    v /= np.linalg.norm(v) 
#    v = np.array([np.sqrt(2)]*2)
#    v = np.array([1.,0])

    p = TestPot()
    cpot = CurvaturePotential(p)

    
    print "------------------------------"
    tcpot = TestCPot(cpot, x)
    tcpot.test_potential(v, verbose=True)
    
    
    print ""
    h = tcpot.getHessian(v)
    hnum = tcpot.NumericalHessian(v, eps=1e-2)
    print "hess   "
    print h
    print "numerical hessian"
    print hnum
    
    
    eps = 1e-6
    delta=np.array([eps, 0.])
    e, g = tcpot.getEnergyGradient(v)
    ep, gp = tcpot.getEnergyGradient(v + delta)
    em, gm = tcpot.getEnergyGradient(v - delta)
    print "hess[0,0]", (ep - 2.*e + em) / eps**2
    print "hess[0,0]", (gp - gm) / eps / 2.
    print "gp, gm", gp, gm

    delta=np.array([0., eps])
    e, g = tcpot.getEnergyGradient(v)
    ep, gp = tcpot.getEnergyGradient(v + delta)
    em, gm = tcpot.getEnergyGradient(v - delta)
    print "hess[1,1]", (ep - 2.*e + em) / eps**2
    print "hess[1,1]", (gp - gm) / (2. * eps)
    print "hess[1,1]", (gp - g) / (eps), "first order"
    print "gp, gm", gp, gm

    plot_test_pot_background(tcpot)
    draw_arrow([0.,0], v)
    plt.show()
    
    return
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

    plot_test_pot(x, v)

def draw_arrow(x, v, ax=None, c='k'):
    ax = ax or plt.gca()
    ax.arrow(x[0], x[1], v[0], v[1], fc=c, ec=c)

def test2():
    from pele.optimize._quench import steepest_descent, lbfgs_py
    from pele.utils.hessian import get_smallest_eig
    p = TestPot2()
    x = np.array([0.4, 0.1])
    v = vec_random_ndim(2) 
#    v = np.array([.4,.1])
#    v /= np.linalg.norm(v) 
#    lam = np.array([.3,-.1])
    
    if True:
        p.test_potential(x)

    hess = p.getHessian(x)
    mu, v = get_smallest_eig(hess)

#    plot_test_pot(x, v, lam)

    cpot = CurvaturePotential(p, lam=10.)
    dpot = ModifiedDimer(p)
    
    xvlam = cpot.merge_x_v(x, v)
    
    xvl_list = []
    def callback(coords=None, gradient=None, rms=None, **kwargs):
#        print "coords", coords, rms
#        print "grad", gradient
        xvl_list.append(coords.copy())
    
    
#    ret = steepest_descent(xvlam, dpot, iprint=1, dx=1e-2, nsteps=20, events=[callback])
    ret = lbfgs_py(xvlam, dpot, iprint=1, maxstep=1.51, nsteps=200, events=[callback], maxErise=2.5)
    
    print ret
    
    if False:
        plot_test_pot_background(dpot, v=np.array([1.0,0]), xmax=.2, ymax=.2)
        plt.show()
    
    
    
    plot_test_pot_background(p=p)
    ax = plt.gca()
    for xvl in xvl_list[0:-1:1]:
        x, v = cpot.split_x_v(xvl)
        v = v / np.linalg.norm(v)
        print x, v
        draw_arrow(x, v, ax)
#        plt.scatter(x[0], x[1])
#        draw_arrow(x, lam, ax, c='r')
    
    plt.show()
    
def draw_hessian_vector_field():
    p = TestPot2()
    x0, y0, xgrid, ygrid = plot_test_pot_background(p=p)
    
    vx = []
    vy = []
    for x, y in izip(xgrid.ravel(), ygrid.ravel()):
        xy = np.array([x, y])
        hess = p.getHessian(xy)
        mu, v = get_smallest_eig(hess)
#        v *= .1
        vx.append(v[0])
        vy.append(v[1])
#        draw_arrow(xy, v)
    
    vx = np.array(vx).reshape(xgrid.shape)
    vy = np.array(vy).reshape(xgrid.shape)
#    plt.streamplot(x0, y0, vx, vy, color='k')
    plt.quiver(x0, y0, vx, vy, color='k')
    plt.show()
        
         

    
if __name__ == "__main__":
    test2()
#    draw_hessian_vector_field()