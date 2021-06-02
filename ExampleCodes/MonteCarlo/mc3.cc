// This version uses a vectorized random number generator

#include <cmath> // for exp
#include <iostream> // for cout, endl
#include <cstdlib> // for random
#include "timerstuff.h" // for cycle_count

// MKL has some fantastic vectorized random number generators, but to
// get an even playing field we use this vectorized Fibonacci
// generator that dates back to Cray vector computer days.  It is NOT
// good enough for production calculations.
#include "ranvec.h"

const int NWARM = 1000;  // Number of iterations to equilbrate (aka warm up) population
const int NITER = 10000; // Number of iterations to sample
const int N = 10240;     // Population size

Random rangen;

double drand() {
  return rangen.get();
}

void vrand(int n, double* r) {
  rangen.getv(n, r);
}

void kernel(double& x, double& p, double ran1, double ran2) {
    double xnew = ran1*23.0;
    double pnew = std::exp(-xnew);
    if (pnew > ran2*p) {
        x = xnew;
        p = pnew;
    }
}

int main() {
    double x[N], p[N], r[2*N];

    // Initialize the points
    for (int i=0; i<N; i++) {
        x[i] = drand()*23.0;
        p[i] = std::exp(-x[i]);
    }
    
    std::cout << "Equilbrating ..." << std::endl;
    for (int iter=0; iter<NWARM; iter++) {
        vrand(2*N, r);
        for (int i=0; i<N; i++) {
            kernel(x[i], p[i], r[i], r[i+N]);
        }
    }

    std::cout << "Sampling and measuring performance ..." << std::endl;
    double sum = 0.0;
    double start = cpu_time();
    uint64_t Xstart = cycle_count();
    for (int iter=0; iter<NITER; iter++) {
        vrand(2*N, r);
        //#pragma simd reduction(+: sum)
        for (int i=0; i<N; i++) {
            kernel(x[i], p[i], r[i], r[i+N]);
            sum += x[i];
        }
    }
    uint64_t Xused = cycle_count() - Xstart;
    double used = cpu_time() - start;

    sum /= (NITER*N);
    std::cout.precision(10);
    std::cout << "the integral is " << sum << " over " << NITER*N << " points " << std::endl;

    double cyc = Xused / double(NITER*N);
    double sec =  used / double((NITER+NWARM)) / double(N);

    std::cout << cyc << " cycles per point " << std::endl;
    std::cout << sec << "    sec per point " << std::endl;

    return 0;
}
