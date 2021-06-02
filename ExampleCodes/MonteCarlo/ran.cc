#include <iostream>
#include <algorithm>

#include <cstdio>
using std::fopen;
using std::fwrite;

/// A random number generator (vectorized, portable and soon to be thread-safe)

/// Following Brent 1992, we use a 48-bit generalized Fibonacci generator 
/// \code
///     u[n] = alpha*u[n-r] + beta*u[n-s] mod m 
/// \endcode
/// with alpha=1, beta=7, r=1279, s=861, m=2^48.  Double precision
/// numbers are used to perform exact integer arithmetic.  48-bit
/// because we have 52 bits of mantissa, alpha+1 is 3 bits and 1 bit spare.

class RandomState {
public:
    int cur;
    double u[4423];
};

class Random {
private:
    const int r;
    const int s;
    const double beta;
    int cur;
    double *u;
    unsigned int simple_state;

    void generate() {
        double * RESTRICT ur = u;
        double * RESTRICT us = u+r-s;
        for (int i=0; i<s; i++) {
            double t = ur[i] + beta*us[i];
            ur[i] = t - int(t);
        }
        
        ur = u+s;
        us = u;
        int rs = r-s;
        for (int i=0; i<rs; i++) {
            double t = ur[i] + beta*us[i];
            ur[i] = t - int(t);
        }
        cur = 0;
    };

    inline unsigned int simple() {
        return simple_state = 1103515245U*simple_state + 12345U;
    };

    Random operator=(const Random&);
    Random(const Random&);

public:
    // a) not clear if beta != 1 is an improvement.
    // b) must ensure s >= r/2.
    // c) r=19937, s=10095 or r=1279, s=861 seem worse than 4423/3004 ?
    //Random(unsigned int seed = 5461) : r(4423), s(3004), beta(7.0), cur(0) {
    Random(unsigned int seed = 5461) : r(1279), s(861), beta(7.0), cur(0) {
        u = new double[r];

        // Initialize startup generator
        if ((seed&1) == 0) seed += 1;
        simple_state = seed;
        for (int i=0; i<10*r; i++) simple();

        // Initialize stream with 48 bit values by first generating
        // roughly 52 bit values and then truncating to exactly 48 bits
        double two21 = 2097152.0;
        double two52 = 4503599627370496.0;
        double two24 = 16777216.0;
        double rtwo24 = 5.9604644775390625e-08;
        for (int i=0; i<r; i++) u[i] = double(simple());
        for (int i=0; i<r; i++) u[i] += double(simple())*two21;
        for (int i=0; i<r; i++) u[i] /= two52;
        // Next line breaks on Cray X1 CC 5.3 ... sigh
        //for (int i=0; i<r; i++) u[i] -= int(u[i]);
        for (int i=0; i<r; i++) {int tmp=int(u[i]); u[i] -= double(tmp);}
        for (int i=0; i<r; i++) {
            int high = int(two24*u[i]);
            int lo = int(two24*(two24*u[i]-high));
            u[i] = rtwo24*(high + rtwo24*lo);
        }

        // Verify that we have only set 48 bits and that at least
        // some of the 48th bits are set.
        double n48 = 0;
        for (int i=0; i<r; i++) {
            double rem = u[i]*two24;
            rem -= int(rem);
            double rem48 = rem*two24;
            rem48 -= int(rem48);
            if (rem48 != 0) {std::cout << "bad bits ?" << std::endl; std::exit(1);}

            double rem47 = rem*two24*0.5;
            rem47 -= int(rem47);
            if (rem47 != 0) n48++;
        }
        if (n48 == 0) {std::cout << "48th bit bad?" << std::endl; std::exit(1);}

        // Warm up
        for (int i=0; i<2000; i++) generate();
    };
    ~Random() {delete [] u;}; 

    inline double get() {
        if (cur >= r) generate();
        return u[cur++];
    };

    void getv(int n, double * RESTRICT v) {
        while (n) {
            if (cur >= r) generate();
            int ndo = std::min(n,r-cur);
            for (int i=0; i<ndo; i++) v[i] = u[i+cur];
            n -= ndo;
            v += ndo;
            cur += ndo;
        }
    };

    void getbytes(int n, unsigned char * RESTRICT v) {
        // This will be much slower than a custom bitstream generator
        // and is presently only used for testing with diehard.
        const double two24 = 16777216.0;
        int n6 = n%6;
        n -= n6;
        while (n>0) {
            if (cur >= r) generate();
            int ndo = std::min(n/6,r-cur);
            for (int i=0; i<ndo; i++) {
                unsigned int high = int(two24*u[i+cur]);
                unsigned int lo = int(two24*(two24*u[i+cur]-high));
                *v++ = (high>>16)&0xff;
                *v++ = (high>>8 )&0xff;
                *v++ = (high    )&0xff;
                *v++ = (lo>>16 )&0xff;
                *v++ = (lo>>8  )&0xff;
                *v++ = (lo     )&0xff;
            };
            n -= ndo*6;
            cur += ndo;
        }
        for (int i=0; i<n6; i++) *v++ = (unsigned char) (256*get());
    };

    void getbytes2(int n, unsigned char * RESTRICT v) {
        while (n) {
            if (cur >= r) generate();
            int ndo = std::min(n,r-cur);
            for (int i=0; i<ndo; i++) v[i] = (unsigned char) (256.0*u[i+cur]);
            n -= ndo;
            v += ndo;
            cur += ndo;
        }
    };

    RandomState getstate() {
        RandomState s;
        s.cur = cur;
        for (int i=0; i<r; i++) s.u[i] = u[i];
        return s;
    };

    void setstate(const RandomState &s) {
        cur = s.cur;
        for (int i=0; i<r; i++) u[i] = s.u[i];
    };
};

int testran() {
    // Crude tests just to find gross errors ... the file written
    // at the end is used as input to diehard.
    Random r;
    double sum1 = 0;
    double sum2 = 0;
    double x12 = 0;
    double two24 = 16777216.0;

    double n = 10000000000.0;
    double nn = n;
    const int nbuf = 64;
    nn /= nbuf;
    n = nn*nbuf;

    double buf[nbuf];

    while (nn--) {
        r.getv(nbuf,buf);
        for (int i=0; i<nbuf; i++) {
          double d1 = buf[i];
          sum1 += d1;
          double d2 = d1 * two24;
          d2 -= int(d2);
          sum2 += d2;
          x12 += (d1-0.5)*(d2-0.5);
       }
    }

    std::cout << "high   " << sum1/n << std::endl;
    std::cout << "lo     " << sum2/n << std::endl;
    std::cout << "hi-lo  " << x12/n << std::endl;

    const int nb = 100000000;
    unsigned char *b = new unsigned char[nb];

    b[nb-1] = 0;
    b[nb] = 99;
    r.getbytes2(nb,b);

    std::cout << int(b[nb-1]) << std::endl;
    std::cout << int(b[nb]) << std::endl;

    FILE *f = fopen("stream","w+");
    if (!f) {std::cout << "fopen?\n"; std::exit(1);}
    fwrite(b, 1, nb, f);
    fclose(f);

    return 0;
}
