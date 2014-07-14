#include <vector>
#include <cstddef>
#include <cassert>
#include "ctime"
#include <memory>
#include <iostream>


using std::vector;
using std::cout;

template<class dtype>
class MyArray1 {
public:
    vector<double> v;

    inline dtype *data() { return v.data(); }
    inline dtype const *data() const { return v.data(); }

    inline dtype &operator[](const size_t i) { return data()[i]; }
    inline dtype const & operator[](const size_t i) const { return data()[i]; }

};
template<class dtype>
class MyArray2 {
public:
    vector<double> v;

    inline dtype *data() { return v.data(); }
    inline dtype const *data() const { return v.data(); }

    inline dtype &operator[](const size_t i) { return data()[i]; }
    inline dtype const operator[](const size_t i) const { return data()[i]; }

};

template<class A>
void test()
{
    static size_t const N_= 1000;
    size_t const N = N_;
    A v;
    v.v.resize(N);
    assert(v.v.size() == N);

    double data[N_];

    clock_t t0 = clock();
    for (size_t k=0; k<10000; ++k) {
        for (size_t i=0; i<N; ++i) {
            v[i] = 1;
        }
        for (size_t i=0; i<N; ++i) {
            data[i] = v[i];
        }
    }
    cout << double(clock() - t0) / CLOCKS_PER_SEC << "\n";
}

int main()
{
//    static size_t const N_= 200;
//    size_t const N = N_;
//    MyArray1<double> v;
//    v.v.resize(N);
//    assert(v.v.size() == N);
//
//    double data[N_];
//
//    clock_t t0 = clock();
//    for (size_t k=0; k<10000; ++k) {
//        for (size_t i=0; i<N; ++i) {
//            v[i] = 1;
//        }
//        for (size_t i=0; i<N; ++i) {
//            data[i] = v[i];
//        }
//    }
//    cout << double(clock() - t0) / CLOCKS_PER_SEC << "\n";
    test<MyArray1<double> >();
    test<MyArray2<double> >();


}
