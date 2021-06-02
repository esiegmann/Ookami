#include <chrono>
#include <cstdint>
#include <stdint.h>
#include <ctime>
#include <sys/time.h>
#include <papi.h>

#define HAVE_PAPI

// Returns current cycle count for this thread
static inline uint64_t cycle_count() {
  return PAPI_get_real_cyc()*18; // <<<<<<<<<<<<<<<<<<<<< 
}

// Returns wall time in seconds from arbitrary origin, accurate to circa a few us.
static inline double wall_time() {
    static bool first_call = true;
    static double start_time;
    
    struct timeval tv;
    gettimeofday(&tv,0);
    double now = tv.tv_sec + 1e-6*tv.tv_usec;
    
    if (first_call) {
        first_call = false;
        start_time = now;
    }
    return now - start_time;
}

// Returns estimate of the cpu frequency.
static double cpu_frequency() {
  return 1.8e9;
}

    static inline double cpu_time() {
#if defined(HAVE_PAPI)
      return PAPI_get_real_usec()*1e-6;
#else
        const auto now = std::chrono::steady_clock::now();
        const auto seconds_since_epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
        return seconds_since_epoch;
#endif
    }
