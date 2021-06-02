#include <iostream>
#include <unistd.h>
#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <omp.h>

const size_t GATHERVL = 16;
const size_t N = 84*GATHERVL; // must be multiple of GATHERVL and 3*N*8 fit into L1
const size_t NREPEAT = 100000;
const size_t MAXTHREAD = 36;

#define ALIGNAS alignas(256)

// These are copied into a thread's stack before use
std::vector<size_t> globalmap(N);
std::vector<size_t> shortmap(N);

void Xsimple(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
  for (size_t i=0; i<n; i++) y[i] = 2.0*x[i] + 3.0*x[i]*x[i];
}

void Xpredicate(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
  for (size_t i=0; i<n; i++) if (x[i]>0) y[i] = x[i];
}

void Xrecip(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
  for (size_t i=0; i<n; i++) y[i] = 1.0/x[i];
}

void Xsqrt(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
  for (size_t i=0; i<n; i++) y[i] = std::sqrt(x[i]);
}

void Xexp(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
  for (size_t i=0; i<n; i++) y[i] = std::exp(x[i]);
}

void Xsin(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
  for (size_t i=0; i<n; i++) y[i] = std::sin(x[i]);
}

void Xpow(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
  for (size_t i=0; i<n; i++) y[i] = std::pow(x[i],0.55);
}

void Xgather(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
#pragma omp simd
  for (size_t i=0; i<n; i++) y[i] = x[map[i]];
}

void Xscatter(size_t n, const size_t* __restrict__ map, const double* __restrict__ x, double* __restrict__ y) {
#pragma omp simd
  for (size_t i=0; i<n; i++) y[map[i]] = x[i];
}

double timer(void(*f)(size_t, const size_t* __restrict__, const double* __restrict__, double* __restrict__)) {
  ALIGNAS double x[N], y[N];
  ALIGNAS size_t map[N];
  for (size_t i=0; i<N; i++) {x[i] = 1.0; y[i] = 1.0; map[i]=globalmap[i];}

  double used = omp_get_wtime();
  for (size_t repeat=0; repeat<NREPEAT; repeat++) {
    f(N, map, x, y);
    asm volatile("" ::: "memory");  // to stop loop reordering
  }
  used = omp_get_wtime() - used;
  if (y[10] > 1e99) std::cout << y[10] << std::endl; // To ensure loop is not completely optimized away
  return used/(N*NREPEAT);
}

std::vector<double> timeromp(void(*f)(size_t n, const size_t* __restrict__,  const double* __restrict__, double* __restrict__)) {
  std::vector<double> result(MAXTHREAD+1);

  for (int nthreads=1; nthreads<=MAXTHREAD; nthreads++) {
    double used = 0.0;
    omp_set_num_threads(nthreads);

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      ALIGNAS double x[N], y[N];
      ALIGNAS size_t map[N];
      for (size_t i=0; i<N; i++) {x[i] = 1.0; y[i] = 1.0; map[i]=globalmap[i];}
      
#pragma omp barrier
      if (tid == 0) used = omp_get_wtime();
      for (size_t repeat=0; repeat<NREPEAT; repeat++) {
	f(N, map, x, y);
	asm volatile("" ::: "memory");  // to stop loop reordering
      }
#pragma omp barrier
      if (tid == 0) used = omp_get_wtime() - used;
      
      if (y[10] > 1e99) std::cout << y[10] << std::endl; // To ensure loop is not completely optimized away
    }
    result[nthreads] = used/(N*NREPEAT);
  }
  return result;
}

int main() {
  std::random_device rdev;
  std::mt19937 rgen(rdev());
  std::uniform_int_distribution<size_t> idist(0,N-1);
  std::uniform_int_distribution<size_t> idist8(0,GATHERVL-1);

  // Make permutation arrays to test gather/scatter
  for (size_t i=0; i<N; i++) globalmap[i] = shortmap[i] = i;
  for (size_t i=0; i<N; i++) std::swap(globalmap[i],globalmap[idist(rgen)]);

  for (size_t i=0; i<N; i+=GATHERVL)
    for (size_t j=0; j<GATHERVL; j++)
      std::swap(shortmap[i+j],shortmap[i+idist8(rgen)]);
			       
  std::vector<std::pair<decltype(Xsimple)*,std::string>> kernels =
    {
     {&Xsimple,"simple"},
     {&Xgather,"gather"},
     {&Xscatter,"scatter"},
     {&Xgather,"shortgather"},
     {&Xscatter,"shortscatter"},
     {&Xpredicate,"predicate"},
     {&Xrecip,"reciprocal"},
     {&Xsqrt,"sqrt"},
     {&Xexp,"exp"},
     {&Xsin,"sin"},
     {&Xpow,"pow"},
    };

  std::vector<std::pair<double,std::vector<double>>> times;
  
  //for (const auto & [kernel,name] : kernels) { // sigh --- Cray c++14 only
  for (const auto& pair : kernels) {
    const auto& kernel = pair.first;
    const auto& name = pair.second;

    if (name == "shortgather" or name == "shortscatter") std::swap(globalmap,shortmap);
    sleep(1); // to let the processor cool down so turbo mode can kick in
    double s = timer(kernel);  // single thread
    auto r = timeromp(kernel); // multiple threads with openmp
    times.push_back({s,r});
    if (name == "shortgather" or name == "shortscatter") std::swap(globalmap,shortmap);
  }

  printf("# threads");
  for (const auto& pair : kernels) {
    const auto& name = pair.second;
    printf("%15s  ", name.c_str());
  }
  printf("\n");
  printf("# -------");
  for (size_t t=1; t<=times.size(); t++) printf("  ---------------");
  printf("\n");

  printf("%6d    ", 0);
  for (const auto& pair : times) printf("    %10.2e   ",pair.first);
  printf("\n");
    
  for (size_t t=1; t<=MAXTHREAD; t++) {
    printf("%6lu    ", t);
    for (const auto& pair : times) printf("    %10.2e   ",pair.second.at(t));
    printf("\n");
  }
  
  return 0;
}
