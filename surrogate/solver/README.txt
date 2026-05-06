This subdirectory contains the fPDE Solver that simulates water and mass transport in fractal-fractured porous media

#### `fractal-fract-PC.cpp`
The fPDE solver for a coupled pressure (P) and concentration (C) system with Caputo fractional derivatives in time (order α) and space (order β) along with a mixed time-space fractional derivative. Uses finite difference technique with Picard iteration, short-memory approximation for fractional derivatives, and a tridiagonal solver all accelerated via OpenCL.

#### `mittag-leffler.h`
Implements the numerical evaluation of the Mittag-Leffler function.

#### `opencl_base.cpp`
Provides C++ wrapper classes for OpenCL operations.

#### `opencl_class.h`
Header file declaring the OpenCL wrapper classes used by `opencl_base.cpp`.

#### `nvidia_pcr.h`
Contains NVIDIA's parallel cyclic reduction (PCR) GPU kernel code (as a C string), implementing a fast parallel tridiagonal system solver in double precision. Modification of the original code from NVIDIA and UC Davis (2009).

