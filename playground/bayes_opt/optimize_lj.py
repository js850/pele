import numpy as np

from pele.systems import LJCluster
from pele.systems.basesystem import dict_copy_update
from pele.takestep import RandomDisplacement

class LJClusterWrap(LJCluster):
    def get_takestep(self, **kwargs):
        kwargs = dict_copy_update(self.params["takestep"], kwargs)
        try:
            stepsize = kwargs.pop("stepsize")
        except KeyError:
            stepsize = 0.6
        print "using stepsize", stepsize

        displace = RandomDisplacement(stepsize=stepsize)
        return displace


def run(system, max_n_quenches=1000000):
    egmin = system.egmin
    delta_e = 1e-2
    
    
    db = system.create_database()
    bh = system.get_basinhopping(db, max_n_minima=1)
    bh.setPrinting(frq=100)
    
    for i in xrange(max_n_quenches):
        bh.run(1)
        emin = db.get_lowest_energy_minimum().energy
        if emin < egmin + delta_e:
            break
        if i == max_n_quenches - 1:
            print "it appears that the run failed. Stopping after", bh.result.nfev, "function evaluations"
    return bh.result.nfev
        

def start(natoms=17, stepsize=0.4, temperature=1., nruns=1, max_n_quenches=1000000):
    gmin_dict = {17 : -61.3180,
            38 : -173.928427,
            }
    system = LJClusterWrap(natoms)
    egmin = gmin_dict[natoms]
    system.egmin = egmin
    
    print "starting job with:"
    print "stepsize", stepsize
    print "temperature", temperature
    print "nrus", nruns
    print "max_n_quenches", max_n_quenches
    
    system.params.takestep.stepsize = stepsize
    system.params.basinhopping.temperature = temperature
    nfev = [run(system, max_n_quenches) for i in xrange(nruns)]
    mean_nfev = np.mean(nfev)
    print "gmin found after", nfev, "evaluations"
    print "return value", mean_nfev
    return mean_nfev

def main(jobid, params):
    return start(natoms=17, stepsize=params["stepsize"], temperature=params["T"])

if __name__ == "__main__":
    main(0, dict(stepsize=.3, T=2.))
