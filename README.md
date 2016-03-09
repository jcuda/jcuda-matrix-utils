# jcuda-matrix-utils

Utility classes for dense and sparse matrices in JCuda.

# NOTE: 

These classes should not be considered to be a stable, general matrix library.
Originally, these classes had been created for one of the 
[JCuda samples](http://www.jcuda.org/samples), namely the example 
implementation of a CG solver based on JCublas and JCusparse.
 
The goal of these classes was to ease the conversion between dense and 
sparse matrices (in CSR format) and for handling the conversion between 
matrices in host-and device memory.

The [samples package](https://github.com/jcuda/jcuda-matrix-utils/blob/master/JCudaMatrixUtils/src/test/java/org/jcuda/matrix/samples)
contains basic examples showing the implementation of CG solvers, and how
to perform a DGEMM operation with sparse matrices using JCusparse.

