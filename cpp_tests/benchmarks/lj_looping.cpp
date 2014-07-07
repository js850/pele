#include "ctime"
#include <memory>

#include "pele/lj.h"
#include "pele/array.h"
#include "pele/lbfgs.h"

using std::cout;
using std::shared_ptr;
using std::make_shared;
using pele::Array;
using pele::BasePotential;
using pele::lj_interaction;
using pele::cartesian_distance;

template<typename pairwise_interaction,
             typename distance_policy = cartesian_distance<3>>
class SimplePairwisePotential1 : public BasePotential
{
public:
    static const size_t _ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> _interaction;
    std::shared_ptr<distance_policy> _dist;

    SimplePairwisePotential1(
            std::shared_ptr<pairwise_interaction> interaction,
            std::shared_ptr<distance_policy> dist=NULL)
        : _interaction(interaction), _dist(dist)
    {}

    inline double get_energy_gradient(Array<double> x, Array<double> grad)
    {
        double e=0.;
        double gij;
        double dr[_ndim];
        const size_t natoms = x.size() / _ndim;

        grad.assign(0.);

        for(size_t atomi=0; atomi<natoms; ++atomi) {
            int i1 = _ndim*atomi;
            for(size_t atomj=0; atomj<atomi; ++atomj) {
                int j1 = _ndim*atomj;

                _dist->get_rij(dr, &x[i1], &x[j1]);

                double r2 = 0;
                for (size_t k=0;k<_ndim;++k){r2 += dr[k]*dr[k];}
                e += _interaction->energy_gradient(r2, &gij, atomi, atomj);
                for(size_t k=0; k<_ndim; ++k)
                    grad[i1+k] -= gij * dr[k];
                for(size_t k=0; k<_ndim; ++k)
                    grad[j1+k] += gij * dr[k];
            }
        }
        return e;
    }
};


int main()
{
    pele::LJ lj(1., 1.);

    size_t natoms = 100;
    pele::Array<double> x(3*natoms);
    for (size_t i=0; i<x.size(); ++i){
        x[i] = (double) i / 2;
    }

    pele::LBFGS lbfgs((pele::BasePotential *) &lj, x);
    lbfgs.run();
    x= lbfgs.get_x();

    auto g = x.copy();
    clock_t t0 = clock();
    for (size_t i = 0; i<1000; ++i){
        lj.get_energy_gradient(x, g);
    }
    cout << double(clock() - t0) / CLOCKS_PER_SEC << "\n";

    auto pot1 = SimplePairwisePotential1<lj_interaction, cartesian_distance<3> >(
            std::make_shared<lj_interaction>(1., 1.), make_shared<cartesian_distance<3> >());
    for (size_t i = 0; i<1000; ++i){
        pot1.get_energy_gradient(x, g);
    }

}
