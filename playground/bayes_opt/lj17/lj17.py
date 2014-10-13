import numpy as np

from playground.bayes_opt.optimize_lj import start

def main(jobid, params):
    return start(natoms=17, stepsize=params["stepsize"], temperature=params["T"], nruns=3)

if __name__ == "__main__":
    main(0, dict(stepsize=.3, T=2.))
