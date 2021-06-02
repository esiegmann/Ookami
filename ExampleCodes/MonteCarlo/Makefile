
TARGETS = exptest mc0 mc1 mc2 mc3 mc4

RESTRICT = -DRESTRICT=__restrict__ 

# A64FX ##############################################
PAPI = -I/opt/cray/pe/papi/6.0.0.4/include -L/opt/cray/pe/papi/6.0.0.4/lib -lpapi

CXX = armclang++
CXXFLAGS = -Ofast -ffp-contract=fast -ffast-math -Wall  -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -march=armv8.2-a+sve -I. -std=c++17 -armpl=sve -mcpu=a64fx -mtune=a64fx  -I/opt/arm/armpl-19.2.0_Cortex-A72_Ubuntu-16.04_arm-hpc-compiler_19.2_aarch64-linux/include  $(PAPI) $(RESTRICT)

#-L/lustre/projects/global/software/a64fx/sleef/3.5.1/lib64/ -lsleef
#-msve-vector-bits=512  ... no impact?
#-mllvm -force-vector-width=8 ... slightly slower code

# gcc 10.2.1 on a64fx
#CXX = g++
#CXXFLAGS = -Ofast -ffast-math -Wall -mtune=a64fx -mcpu=a64fx  -march=armv8.2-a+sve -fopt-info-vec -fopt-info-vec-missed $(PAPI) $(RESTRICT)
#-L/lustre/projects/global/software/a64fx/sleef/3.5.1/lib64/ -lsleef
# slower --param aarch64-autovec-preference=2 -msve-vector-bits=512

# gcc 10.2.1 on a64fx with glibc 2.32 --- BROKEN
#SYSROOT=/lustre/projects/global/software/a64fx/glibc/2.32
#GLIBCLD=-L/lustre/projects/global/software/a64fx/glibc/2.32/usr/lib64 -Wl,-rpath=/lustre/projects/global/software/a64fx/glibc/2.32/lib64 -Wl,--dynamic-linker=/lustre/projects/global/software/a64fx/glibc/2.32/lib64/ld-2.32.so
#GLIBCCX=-I/lustre/projects/global/software/a64fx/glibc/2.32/include
#CXX = g++
#CXXFLAGS = -Ofast -ffast-math -Wall -mtune=a64fx -mcpu=a64fx  -march=armv8.2-a+sve -fopt-info-vec -fopt-info-vec-missed $(PAPI) $(RESTRICT) $(GLIBCLD) $(GLIBCCX)

#-L/lustre/projects/global/software/a64fx/sleef/3.5.1/lib64/ -lsleef
# slower --param aarch64-autovec-preference=2 -msve-vector-bits=512

# gcc 9.3.0 from ARM on a64fx
#CXX = g++
#CXXFLAGS = -Ofast -ffast-math -Wall -mtune=native -mcpu=native  -march=armv8.2-a+sve -fopt-info-vec -fopt-info-vec-missed $(PAPI) $(RESTRICT)

# Cray CC 10.0.1
#CXX = CC
#CXXFLAGS = -O3 -h aggress,flex_mp=tolerant,msgs,negmsgs,vector3 $(PAPI) $(RESTRICT)
#preferred_vector_width=512


# X86 ##############################################
#MKLROOT = /opt/intel/mkl
#LDFLAGS = -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

#CXX = icpc on x86
#CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -opt-report -mkl=sequential

#CXX = g++-9 on x86
# try Ofast, O3, O2, O1
#CXXFLAGS = -Ofast -march=native -fopt-info-vec-optimized -m64 -I$(MKLROOT)/include

all:	$(TARGETS)

vexp.o:	vexp.cc
	g++ -c -Ofast -ffast-math -Wall -mtune=native -mcpu=native -std=c++17 -march=armv8.2-a+sve -msve-vector-bits=512 vexp.cc 
	#g++ -c -Ofast -ffast-math -Wall -mtune=a64fx -mcpu=a64fx -std=c++17 -march=armv8.2-a+sve -msve-vector-bits=512 vexp.cc

%.o : %.cc timerstuff.h Makefile ranvec.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~ *optrpt


mc4:	mc4.o vexp.o
	$(CXX) $(CXXFLAGS) $< vexp.o -o $@


exptest:	exptest.o vexp.o
	$(CXX) $(CXXFLAGS) $< vexp.o -o $@
