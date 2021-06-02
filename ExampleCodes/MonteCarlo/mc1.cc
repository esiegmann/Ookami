// This version switches the loop orders and introduces two vectors in
// preparation for vectorizing the loops

#include <cmath> // for exp
#include <iostream> // for cout, endl
#include <cstdlib> // for random
#include "timerstuff.h" // for cycle_count

const int NWARM = 1000;  // Number of iterations to equilbrate (aka warm up) population
const int NITER = 10000; // Number of iterations to sample
const int N = 10240;     // Population size

double drand() {
    const double fac = 1.0/(RAND_MAX-1.0);
    return fac*random();
}

void kernel(double& x, double& p) {
    double xnew = drand()*23.0;
    double pnew = std::exp(-xnew);
    if (pnew > drand()*p) {
        x = xnew;
        p = pnew;
    }
}

int main() {
    double x[N], p[N];

    // Initialize the points
    for (int i=0; i<N; i++) {
        x[i] = drand()*23.0;
        p[i] = std::exp(-x[i]);
    }
    
    std::cout << "Equilbrating ..." << std::endl;
    for (int iter=0; iter<NWARM; iter++) {
        for (int i=0; i<N; i++) {
            kernel(x[i], p[i]);
        }
    }

    std::cout << "Sampling and measuring performance ..." << std::endl;
    double sum = 0.0;
    double start = cpu_time();
    uint64_t Xstart = cycle_count();
    for (int iter=0; iter<NITER; iter++) {
        for (int i=0; i<N; i++) {
            kernel(x[i], p[i]);
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
