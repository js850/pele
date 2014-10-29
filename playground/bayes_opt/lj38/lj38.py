import numpy as np

from playground.bayes_opt.optimize_lj import start

def main(jobid, params):
    nfev = start(natoms=38, stepsize=params["stepsize"], temperature=params["T"], nruns=2, max_n_quenches=10000)
    return np.log(nfev)

if __name__ == "__main__":
    main(0, dict(stepsize=.9, T=2.))
