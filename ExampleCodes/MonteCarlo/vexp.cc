#include <arm_sve.h>
#include <cmath>

// SIMD vector types 
#define F64 svfloat64_t
#define I64 svint64_t
#define U64 svuint64_t
#define MASK svbool_t

// Mask values depending on vector length
#define EVERYTHING svptrue_b64()

// FP ceil, floor, round operations
#define CEIL(mask,v) svrintp_x(mask,v)
#define ROUND(mask,v) svrinta_x(mask,v)
#define FLOOR(mask,v) svrintm_x(mask,v)

// FP convert to integer
#define INT(mask,v) svcvt_s64_x(mask, v)
#define UINT(mask,v) svcvt_u64_x(mask, v)

// Integer convert to FP
#define FLOAT(mask,v) svcvt_f64_x(mask, v)

// Integer shift to right rounding to -infinity (int(floor(value/2**shift)))
// shift can be immediate value or vector of values
#define ASR(mask,v,shift) svasr_x(pg,v,shift)

// Duplicate scalar across all elements
#define IDUP(value) svdup_s64(value)
#define FDUP(value) svdup_f64(value)

// Load and store
#define LOAD(mask,ptr) svld1(mask, ptr)
#define STORE(mask,ptr,vec) svstnt1(mask, ptr, vec); 

// result = a*b + c --- is there a constraint on result?
#define FMA(mask,a,b,c) svmad_f64_x(mask,a,b,c)
#define IMA(mask,a,b,c) svmad_s64_x(mask,a,b,c)

// result = a*b
#define MUL(mask,a,b) svmul_x(mask,a,b)

// Vector exponential with VL=8 and no unrolling
void vexp(int64_t n, const double* __restrict__ xvec, double* __restrict__ yvec) {
  static const double fac = -0.0108304246962491454596442518978; // -log2/64
  static const double rfac = 92.3324826168936580710351795840;

  // Chebyshev on [-a,a] a=log(2)/128
  static const double a0 = 1.00000000000000000109448766559;
  static const double a1 = 1.00000000000000000054724376115;
  static const double a2 = 0.499999999999328180895493906552;
  static const double a3 = 0.166666666666517373549704816583;
  static const double a4 = 0.0416667277594639384346492115235;
  static const double a5 = 0.00833334351546532331159118269769;

  // // Cheb on [-a,a] for cubic and higher but exact Taylor coeffs on lower
  // static const double a0 = 1.0;
  // static const double a1 = 1.0;
  // static const double a2 = 0.5;
  // static const double a3 = 0.166666666666645339082562230955;
  // static const double a4 = 0.0416666972130599706546300218462;    
  // static const double a5 = 0.00833333915169364528960093698321;

  F64 vfac = FDUP(fac);
  F64 vrfac = FDUP(rfac);
  F64 va0 = FDUP(a0);
  F64 va1 = FDUP(a1);
  F64 va2 = FDUP(a2);
  F64 va3 = FDUP(a3);
  F64 va4 = FDUP(a4);
  F64 va5 = FDUP(a5);
  I64 v1023 = IDUP((int64_t(1023)<<6));

  int64_t rem = n&7l;
  int64_t n8 = n-rem;
  MASK pg = EVERYTHING;

#define BODY \
      F64 vx = LOAD(pg, xvec+j); \
      F64 vdk = ROUND(pg, MUL(pg, vx, vrfac));  \
      I64 vk = INT(pg, vdk); \
      F64 vr = FMA(pg,vdk,vfac,vx); \
      F64 vr2 = MUL(pg,vr,vr); \
      F64 vr45 = FMA(pg,va5,vr,va4); \
      F64 vr23 = FMA(pg,va3,vr,va2); \
      F64 vr01 = FMA(pg,va1,vr,va0); \
      F64 vr2345 = FMA(pg,vr45,vr2,vr23); \
      F64 vr012345 = FMA(pg,vr2345,vr2,vr01); \
      vk = svadd_x(pg, vk, v1023); \
      F64 vexpa = svexpa_f64(svreinterpret_u64(vk)); \
      STORE(pg, yvec+j, MUL(pg,vexpa,vr012345))
  
  for (int64_t j=0; j<n8; j+=8) {BODY;}
  if (rem) {
      int64_t j = n8;
      MASK pg = svwhilelt_b64(j, n);
      BODY;
  }
  
}

// Vector exponential with VLA and no unrolling
void vexp_varloop(int64_t n, const double* __restrict__ xvec, double* __restrict__ yvec) {
  static const double fac = -0.0108304246962491454596442518978; // -log2/64
  static const double rfac = 92.3324826168936580710351795840;

  // Chebyshev on [-a,a] a=log(2)/128
  static const double a0 = 1.00000000000000000109448766559;
  static const double a1 = 1.00000000000000000054724376115;
  static const double a2 = 0.499999999999328180895493906552;
  static const double a3 = 0.166666666666517373549704816583;
  static const double a4 = 0.0416667277594639384346492115235;
  static const double a5 = 0.00833334351546532331159118269769;

  F64 vfac = FDUP(fac);
  F64 vrfac = FDUP(rfac);
  F64 va0 = FDUP(a0);
  F64 va1 = FDUP(a1);
  F64 va2 = FDUP(a2);
  F64 va3 = FDUP(a3);
  F64 va4 = FDUP(a4);
  F64 va5 = FDUP(a5);
  I64 v1023 = IDUP((int64_t(1023)<<6));

  int64_t j;
  MASK pg;
  for (j=0, pg=svwhilelt_b64(j, n);
       svptest_any(svptrue_b64(),pg);
       j+=svcntd(), pg=svwhilelt_b64(j,n)) {
    BODY;
  }  
}

// Vector exponential with VL=8 unroll=2 and either Horner or Estrin polyn evaluation
void vexpu2(int64_t n, const double* __restrict__ xvec, double* __restrict__ yvec) {
  static const double fac = -0.0108304246962491454596442518978; // -log2/64
  static const double rfac = 92.3324826168936580710351795840;

  // Chebyshev on [-a,a] a=log(2)/128
  static const double a0 = 1.00000000000000000109448766559;
  static const double a1 = 1.00000000000000000054724376115;
  static const double a2 = 0.499999999999328180895493906552;
  static const double a3 = 0.166666666666517373549704816583;
  static const double a4 = 0.0416667277594639384346492115235;
  static const double a5 = 0.00833334351546532331159118269769;

  // // Cheb on [-a,a] for cubic and higher but exact Taylor coeffs on lower
  // static const double a0 = 1.0;
  // static const double a1 = 1.0;
  // static const double a2 = 0.5;
  // static const double a3 = 0.166666666666645339082562230955;
  // static const double a4 = 0.0416666972130599706546300218462;    
  // static const double a5 = 0.00833333915169364528960093698321;

  F64 vfac = FDUP(fac);
  F64 vrfac = FDUP(rfac);
  F64 va0 = FDUP(a0);
  F64 va1 = FDUP(a1);
  F64 va2 = FDUP(a2);
  F64 va3 = FDUP(a3);
  F64 va4 = FDUP(a4);
  F64 va5 = FDUP(a5);
  I64 v1023 = IDUP((int64_t(1023)<<6));

  int64_t rem = n&15l;
  int64_t n16 = n-rem;
  MASK pg = EVERYTHING;

#define BODY \
      F64 vx = LOAD(pg, xvec+j); \
      F64 vdk = ROUND(pg, MUL(pg, vx, vrfac));  \
      I64 vk = INT(pg, vdk); \
      F64 vr = FMA(pg,vdk,vfac,vx); \
      F64 vr2 = MUL(pg,vr,vr); \
      F64 vr45 = FMA(pg,va5,vr,va4); \
      F64 vr23 = FMA(pg,va3,vr,va2); \
      F64 vr01 = FMA(pg,va1,vr,va0); \
      F64 vr2345 = FMA(pg,vr45,vr2,vr23); \
      F64 vr012345 = FMA(pg,vr2345,vr2,vr01); \
      vk = svadd_x(pg, vk, v1023); \
      F64 vexpa = svexpa_f64(svreinterpret_u64(vk)); \
      STORE(pg, yvec+j, MUL(pg,vexpa,vr012345))

// Estrin
// #define BODY2							\
//     F64 vx = LOAD(pg, xvec+j);                                  F64 vx_2 = LOAD(pg, xvec+j+8); \
//     F64 vdk = ROUND(pg, MUL(pg, vx, vrfac));                    F64 vdk_2 = ROUND(pg, MUL(pg, vx_2, vrfac)); \
//     I64 vk = INT(pg, vdk);                                      I64 vk_2 = INT(pg, vdk_2); \
//     F64 vr = FMA(pg,vdk,vfac,vx);                               F64 vr_2 = FMA(pg,vdk_2,vfac,vx_2); \
//     F64 vr2 = MUL(pg,vr,vr);                                    F64 vr2_2 = MUL(pg,vr_2,vr_2); \
//     F64 vr45 = FMA(pg,va5,vr,va4);                              F64 vr45_2 = FMA(pg,va5,vr_2,va4); \
//     F64 vr23 = FMA(pg,va3,vr,va2);                              F64 vr23_2 = FMA(pg,va3,vr_2,va2); \
//     F64 vr01 = FMA(pg,va1,vr,va0);                              F64 vr01_2 = FMA(pg,va1,vr_2,va0); \
//     F64 vr2345 = FMA(pg,vr45,vr2,vr23);                         F64 vr2345_2 = FMA(pg,vr45_2,vr2_2,vr23_2); \
//     F64 vr012345 = FMA(pg,vr2345,vr2,vr01);                     F64 vr012345_2 = FMA(pg,vr2345_2,vr2_2,vr01_2); \
//     vk = svadd_x(pg, vk, v1023);                                vk_2 = svadd_x(pg, vk_2, v1023); \
//     F64 vexpa = svexpa_f64(svreinterpret_u64(vk));              F64 vexpa_2 = svexpa_f64(svreinterpret_u64(vk_2)); \
//     STORE(pg, yvec+j, MUL(pg,vexpa,vr012345));                  STORE(pg, yvec+j+8, MUL(pg,vexpa_2,vr012345_2))

// Horner
#define BODY2								\
    F64 vx = LOAD(pg, xvec+j);                                F64 vx_2 = LOAD(pg, xvec+j+8);			    \
    F64 vdk = ROUND(pg, MUL(pg, vx, vrfac));		      F64 vdk_2 = ROUND(pg, MUL(pg, vx_2, vrfac));	    \
    I64 vk = INT(pg, vdk);				      I64 vk_2 = INT(pg, vdk_2);			    \
    F64 vr = FMA(pg,vdk,vfac,vx);			      F64 vr_2 = FMA(pg,vdk_2,vfac,vx_2);		    \
    F64 v = FMA(pg,va5,vr,va4);				      F64 v_2 = FMA(pg,va5,vr_2,va4);			    \
    v = FMA(pg,vr,v,va3);				      v_2 = FMA(pg,vr_2,v_2,va3);			    \
    v = FMA(pg,vr,v,va2);				      v_2 = FMA(pg,vr_2,v_2,va2);			    \
    v = FMA(pg,vr,v,va1);				      v_2 = FMA(pg,vr_2,v_2,va1);			    \
    v = FMA(pg,vr,v,va0);				      v_2 = FMA(pg,vr_2,v_2,va0);			    \
    vk = svadd_x(pg, vk, v1023);			      vk_2 = svadd_x(pg, vk_2, v1023);		    \
    F64 vexpa = svexpa_f64(svreinterpret_u64(vk));	      F64 vexpa_2 = svexpa_f64(svreinterpret_u64(vk_2));  \
    STORE(pg, yvec+j, MUL(pg,vexpa,v));             	      STORE(pg, yvec+j+8, MUL(pg,vexpa_2,v_2)) 
  
  
  for (int64_t j=0; j<n16; j+=16) {BODY2;}
  while (n16<n) {
      int64_t j = n16;
      MASK pg = svwhilelt_b64(j, n);
      BODY;
      n16 += 8;
  }
}
