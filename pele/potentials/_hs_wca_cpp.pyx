"""
# distutils: language = C++
"""
cimport pele.potentials._pele as _pele
import numpy as np
cimport numpy as np
from cpython cimport bool
from ctypes import c_size_t as size_t

# use external c++ class
cdef extern from "pele/hs_wca.h" namespace "pele":
    cdef cppclass  cHS_WCA "pele::HS_WCA":
        cHS_WCA(double eps, double sca, _pele.Array[double] radii) except +
    cdef cppclass  cHS_WCAPeriodic "pele::HS_WCAPeriodic":
        cHS_WCAPeriodic(double eps, double sca, _pele.Array[double] radii, double * boxvec) except +
    cdef cppclass  cHS_WCANeighborList "pele::HS_WCANeighborList":
        cHS_WCANeighborList(_pele.Array[long] & ilist, double eps, double sca, _pele.Array[double] radii) except +    
    cdef cppclass  cHS_WCAFrozen "pele::HS_WCAFrozen":
        cHS_WCAFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +
    cdef cppclass  cHS_WCAPeriodicFrozen "pele::HS_WCAPeriodicFrozen":
        cHS_WCAPeriodicFrozen(double eps, double sca, _pele.Array[double] radii, double* boxvec, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +

cdef class HS_WCA(_pele.BasePotential):
    """define the python interface to the c++ HS_WCA implementation
    """
    cpdef bool periodic 
    #thickness of the wca shell is sca * R where R is the hard core radius of the sphere
    def __cinit__(self, eps, sca, np.ndarray[double, ndim=1] radii, boxvec=None, boxl=None):
        assert not (boxvec is not None and boxl is not None)
        if boxl is not None:
            boxvec = [boxl] * 3
        cdef np.ndarray[double, ndim=1] bv
                
        if boxvec is None:
            self.periodic = False
            self.thisptr = <_pele.cBasePotential*>new cHS_WCA(eps, sca, _pele.Array[double](<double*> radii.data, radii.size))
        else:
            self.periodic = True
            bv = np.array(boxvec, dtype=float)
            self.thisptr = <_pele.cBasePotential*>new cHS_WCAPeriodic(eps, sca, _pele.Array[double](<double*> radii.data, radii.size), <double*> bv.data)

cdef class HS_WCAFrozen(_pele.BasePotential):
    """define the python interface to the c++ HS_WCAFrozen implementation
    """
    cpdef bool periodic
    def __cinit__(self, np.ndarray[double, ndim=1] reference_coords, frozen_atoms, eps, sca, np.ndarray[double, ndim=1] radii, boxvec=None, boxl=None):
        assert not (boxvec is not None and boxl is not None)
        cdef np.ndarray[long, ndim=1] frozen_dof
        frozen_dof = np.array([range(3*i,3*i+3) for i in frozen_atoms], dtype=int).reshape(-1) 
                    
        if boxl is not None:
            boxvec = np.array([boxl] * 3)
        cdef np.ndarray[double, ndim=1] bv
          
        if boxvec is None:
            self.periodic = False
            self.thisptr = <_pele.cBasePotential*>new cHS_WCAFrozen(eps, sca, _pele.Array[double](<double*> radii.data, radii.size), _pele.Array[double](<double *> reference_coords.data, reference_coords.size),
                        _pele.Array[size_t](<size_t *> frozen_dof.data, frozen_dof.size) )
        else:
            self.periodic = True
            bv = np.array(boxvec, dtype=float)
            assert bv.size == 3
            self.thisptr = <_pele.cBasePotential*>new cHS_WCAPeriodicFrozen(eps, sca, _pele.Array[double](<double*> radii.data, radii.size), <double*> bv.data, _pele.Array[double](<double *> reference_coords.data, reference_coords.size),
                        _pele.Array[size_t](<size_t *> frozen_dof.data, frozen_dof.size) )

cdef class HS_WCANeighborList(_pele.BasePotential):
    """define the python interface to the c++ HS_WCA implementation
    """
    def __cinit__(self, np.ndarray[long, ndim=1] ilist, eps, sca, np.ndarray[double, ndim=1] radii):
        self.thisptr = <_pele.cBasePotential*>new cHS_WCANeighborList( _pele.Array[long](<long*> ilist.data, <int> ilist.size), eps, sca, _pele.Array[double](<double*> radii.data, radii.size))

cdef class _ErrorPotential(_pele.BasePotential):
    """this is a test potential which should raise an exception when called
    """
    def __cinit__(self):
        self.thisptr = <_pele.cBasePotential*>new _pele.cBasePotential()
  
