# Ookami
Ookami (https://www.stonybrook.edu/ookami/) is a computer technology testbed supported by the National Science Foundation under grant OAC 1927880. It provides researchers with access to the A64FX processor developed by Riken and Fujitsu for the Japanese path to exascale computing and is currently deployed in the fastest computer in the world, Fugaku. It is the first such computer outside of Japan. By focusing on crucial architectural details, the ARM-based, multi-core, 512-bit SIMD-vector processor with ultrahigh-bandwidth memory promises to retain familiar and successful programming models while achieving very high performance for a wide range of applications. It supports a wide range of data types and enables both HPC and big data applications.

The Ookami HPE (formerly Cray) Apollo 80 system has 174 A64FX compute nodes each with 32GB of high-bandwidth memory and a 512 Gbyte SSD. This amounts to about 1.5M node hours per year. A high-performance Lustre filesystem provides about 0.8Pbyte storage.

To facilitate users exploring current computer technologies and contrasting performance and programmability with the A64FX, Ookami also includes:

1 node with dual socket AMD Rome (128 cores) with 512 Gbyte memory
2 nodes with dual socket Thunder X2 (64 cores) each with 256 Gbyte memory and 2 NVIDIA V100 GPU
Intel Sky Lake Processors (32 cores) with 192 Gbyte memory

This repositiry collects small test programms and results from Ookami.
