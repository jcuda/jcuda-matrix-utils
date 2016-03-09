package org.jcuda.matrix.samples;
/*
 * JCusparse - Java bindings for CUSPARSE, the NVIDIA CUDA sparse
 * matrix library, to be used with JCuda
 *
 * Copyright (c) 2010-2012 Marco Hutter - http://www.jcuda.org
 */

import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDcsrgemm;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.JCusparse.cusparseSetPointerMode;
import static jcuda.jcusparse.JCusparse.cusparseXcsrgemmNnz;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.runtime.JCuda;

import org.jcuda.matrix.d.DoubleMatrices;
import org.jcuda.matrix.d.DoubleMatrixDeviceCSR;
import org.jcuda.matrix.d.DoubleMatrixHostCSR;
import org.jcuda.matrix.d.DoubleMatrixHostDense;


/**
 * A sample application showing how to use JCusparse to perform a DGEMM
 * operation on sparse matrices.
 */
public class JCusparseSampleDgemm
{
    /**
     * The entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String args[])
    {
        // Enable exceptions and subsequently omit error checks in this sample
        JCusparse.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);

        // Create the input matrices for this sample. For convenience, 
        // the matrices are first created as dense matrices and filled
        // with the desired values, and then converted into sparse 
        // matrices in CSR format. (For real applications, the host
        // matrices might already be available in CSR format)
        int rowsA = 3;
        int colsA = 4;
        int rowsB = colsA;
        int colsB = 4;
        int rowsC = rowsA;
        int colsC = colsB;
        double epsilon = 1e-10f;
        
        DoubleMatrixHostDense hostDenseA = 
            DoubleMatrices.createDoubleMatrixHostDense(rowsA, colsA);
        hostDenseA.set(0, 0, 1.0);
        hostDenseA.set(1, 1, 2.0);
        hostDenseA.set(0, 2, 3.0);
        hostDenseA.set(2, 3, 4.0);
        DoubleMatrixHostCSR hostCsrA = 
            DoubleMatrices.createDoubleMatrixHostCSR(hostDenseA, epsilon);
        
        DoubleMatrixHostDense hostDenseB = 
            DoubleMatrices.createDoubleMatrixHostDense(rowsB, colsB);
        hostDenseB.set(1, 0, 1.0);
        hostDenseB.set(0, 1, 2.0);
        hostDenseB.set(3, 1, 3.0);
        hostDenseB.set(1, 2, 4.0);
        hostDenseB.set(3, 2, 5.0);
        hostDenseB.set(0, 3, 6.0);
        hostDenseB.set(2, 3, 7.0);
        DoubleMatrixHostCSR hostCsrB = 
            DoubleMatrices.createDoubleMatrixHostCSR(hostDenseB, epsilon);
        
        // Print the input matrices
        System.out.println("A:\n"+DoubleMatrices.toString2D(hostCsrA));
        System.out.println("B:\n"+DoubleMatrices.toString2D(hostCsrB));
        
        // Create the CUSPARSE handle
        cusparseHandle handle = new cusparseHandle();
        cusparseCreate(handle);

        // Copy the CSR matrices from the host to the device
        DoubleMatrixDeviceCSR deviceCsrA = 
            DoubleMatrices.createDoubleMatrixDeviceCSR(handle, hostCsrA);
        DoubleMatrixDeviceCSR deviceCsrB = 
            DoubleMatrices.createDoubleMatrixDeviceCSR(handle, hostCsrB);

        // Create the matrix descriptor for the result matrix
        cusparseMatDescr matDescrC = new cusparseMatDescr();
        cusparseCreateMatDescr(matDescrC);
        cusparseSetMatType(matDescrC, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(matDescrC, CUSPARSE_INDEX_BASE_ZERO);
        
        // Compute the number of nonzero elements that the result matrix will 
        // have. This is done by calling "cusparseXcsrgemmNnz" with the same 
        // parameters that will later be used for the "cusparseDcsrgemm" call
        CUdeviceptr csrRowPtrC = new CUdeviceptr();
        cudaMalloc(csrRowPtrC, Sizeof.INT*(rowsA+1));
        int transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        int transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        int nnzCArray[] = { 0 };
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
        cusparseXcsrgemmNnz(handle, transA, transB, rowsA, colsB, colsA, 
            deviceCsrA.getDescriptor(), 
            deviceCsrA.getNumNonZeros(), 
            deviceCsrA.getRowPointers(), 
            deviceCsrA.getColumnIndices(),
            deviceCsrB.getDescriptor(), 
            deviceCsrB.getNumNonZeros(), 
            deviceCsrB.getRowPointers(), 
            deviceCsrB.getColumnIndices(),
            matDescrC, csrRowPtrC, Pointer.to(nnzCArray));

        // Compute the number of nonzero elements according to 
        // http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-csrgemm
        int baseArray[] = { 0 };
        cudaMemcpy(Pointer.to(nnzCArray), 
            csrRowPtrC.withByteOffset(rowsA * Sizeof.INT), 
            1 * Sizeof.INT, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(baseArray), csrRowPtrC, 
            1 * Sizeof.INT, cudaMemcpyDeviceToHost);
        int nnzC = nnzCArray[0] - baseArray[0];

        System.out.println("Number of nonzero elements in result: "+nnzC);
        
        // Allocate the memory for the result matrix, based on the number
        // on nonzero elements
        CUdeviceptr csrColIndexC = new CUdeviceptr();
        CUdeviceptr csrValC = new CUdeviceptr();
        cudaMalloc(csrColIndexC, Sizeof.DOUBLE * nnzC);
        cudaMalloc(csrValC, Sizeof.DOUBLE * nnzC);
        
        // Finally, perform the DGEMM
        cusparseDcsrgemm(handle, transA, transB, rowsA, colsB, colsA,
            deviceCsrA.getDescriptor(), 
            deviceCsrA.getNumNonZeros(), 
            deviceCsrA.getValues(), 
            deviceCsrA.getRowPointers(), 
            deviceCsrA.getColumnIndices(),
            deviceCsrB.getDescriptor(), 
            deviceCsrB.getNumNonZeros(), 
            deviceCsrB.getValues(), 
            deviceCsrB.getRowPointers(),
            deviceCsrB.getColumnIndices(),
            matDescrC, csrValC, csrRowPtrC, csrColIndexC);
        
        // For convenience, convert the result into a matrix instance
        // and use the utility function to copy it back to the host,
        DoubleMatrixDeviceCSR deviceCsrC = 
            DoubleMatrices.createDoubleMatrixDeviceCSR(
                matDescrC, rowsC, colsC, nnzC, 
                csrValC, csrRowPtrC, csrColIndexC);
        DoubleMatrixHostCSR hostCsrC = 
            DoubleMatrices.createDoubleMatrixHostCSR(deviceCsrC);
        
        System.out.println("C:\n"+hostCsrC);
        
        // Clean up
        deviceCsrA.dispose();
        deviceCsrB.dispose();
        deviceCsrC.dispose();
        cusparseDestroy(handle);
    }
}
