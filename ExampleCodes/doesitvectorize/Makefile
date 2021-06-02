
# close binding since we want to see effect of loading a single socket
OMPENV = OMP_PLACES=CORES OMP_PROC_BIND=close OMP_NUM_THREADS=48 
RUNCMD = srun -N 1 -p short -c 48

# GCC 10.2.0 or older
gcc-old:
	which g++
	g++ --version
	g++ --std=c++17 -Ofast -ffast-math -Wall -mtune=native -mcpu=native  -march=armv8.2-a+sve -fopt-info-vec -fopt-info-vec-missed -fopenmp test.cc
	$(OMPENV) $(RUNCMD) ./a.out > out.`which g++ | sed -e s,/,:,g`

# GCC 10.2.1 or later understands a64fx target
gcc:
	which g++
	g++ --version
	g++ -Ofast -ffast-math -Wall -mtune=a64fx -mcpu=a64fx  -march=armv8.2-a+sve -fopt-info-vec -fopt-info-vec-missed -fopenmp test.cc
	$(OMPENV) $(RUNCMD) ./a.out > out.`which g++ | sed -e s,/,:,g`

# ARM clang
arm:
	which armclang++
	armclang++ --version
	armclang++ --std=c++17 -Ofast -ffp-contract=fast -ffast-math -Wall  -Rpass=loop-vectorize -march=armv8.2-a+sve -mcpu=a64fx -armpl -fopenmp test.cc
	$(OMPENV) $(RUNCMD) ./a.out > out.`which armclang++ | sed -e s,/,:,g` 

# CRAY
cray:
	which CC
	CC --version
	CC -O3 -h aggress,flex_mp=tolerant,msgs,negmsgs,vector3,omp test.cc
	$(OMPENV) $(RUNCMD) ./a.out > out.`which CC | sed -e s,/,:,g`

# Intel Skylake
intel:
	which icpc
	icpc --version
	icpc -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopt-report=5  -qopt-report-phase=vec -mkl=sequential -qopt-zmm-usage=high -qopenmp test.cc
	$(OMPENV) ./a.out > out.`which icpc | sed -e s,/,:,g`

clean:
	/bin/rm -rf a.out *~
