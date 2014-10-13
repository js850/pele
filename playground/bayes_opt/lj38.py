import numpy as np

from optimize_lj import start

def main(jobid, params):
    return start(natoms=38, stepsize=params["stepsize"], temperature=params["T"], nruns=1, max_n_quenches=10000)

if __name__ == "__main__":
    main(0, dict(stepsize=.9, T=2.))
