/*
 * JCudaMatrixUtils - Matrix utility classes for JCuda
 *
 * Copyright (c) 2010-2016 Marco Hutter - http://www.jcuda.org
 */
package org.jcuda.matrix.samples;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasSetPointerMode;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_HOST;
import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.jcusparse.JCusparse.cusparseSetPointerMode;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST;
import static jcuda.runtime.JCuda.cudaFree;

import java.util.Random;

import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;

import org.jcuda.matrix.Utils;
import org.jcuda.matrix.f.FloatMatrices;
import org.jcuda.matrix.f.FloatMatrixDeviceCSR;
import org.jcuda.matrix.f.FloatMatrixHost;
import org.jcuda.matrix.f.FloatMatrixHostDense;
import org.jcuda.matrix.f.MutableFloatMatrixHost;
import org.jcuda.matrix.f.cg.CGSolverFloatDevice;
import org.jcuda.matrix.f.cg.CGSolversFloat;

/**
 * A sample application solving a system of linear equations using
 * two different implementations of the {@link CGSolverFloatDevice}
 * interface. One implementation implements the CG method directly.
 * The other applies an incomplete Cholesky preconditioner.
 */
public class CGSolverFloatSample
{
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String args[])
    {
        // Enable exceptions and subsequently omit error checks in this sample
        JCuda.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);
        JCusparse.setExceptionsEnabled(true);

        // Create the handle for the CUSPARSE context
        cusparseHandle cusparseHandle = new cusparseHandle();
        cusparseCreate(cusparseHandle);
        cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);

        // Create the handle for the CUBLAS context
        cublasHandle cublasHandle = new cublasHandle();
        cublasCreate(cublasHandle);
        cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);

        // Create the input matrix: A random tridiagonal symmetric dense 
        // matrix that is stored in host memory
        int numRows = 1024;
        int numCols = numRows;
        FloatMatrixHostDense matrixHostDense = createMatrix(numRows, numCols);

        // Create the solver that implements the CG method 
        // and run the test
        CGSolverFloatDevice solverSimple = 
            CGSolversFloat.createSimple(cusparseHandle, cublasHandle);
        runTest(cusparseHandle, matrixHostDense, solverSimple);

        // Create the solver that implements the preconditioned CG method 
        // and run the test
        CGSolverFloatDevice solverPrecond = 
            CGSolversFloat.createPrecond(cusparseHandle, cublasHandle);
        runTest(cusparseHandle, matrixHostDense, solverPrecond);
        
        // Clean up
        solverSimple.dispose();
        solverPrecond.dispose();
        cusparseDestroy(cusparseHandle);
        cublasDestroy(cublasHandle);
    }
    
    /**
     * Run a test, solving a system of linear equations with the given
     * matrix, using the given solver.
     * 
     * @param cusparseHandle The CUSPARSE context handle
     * @param matrixHostDense The input matrix
     * @param solver The solver
     */
    static void runTest(cusparseHandle cusparseHandle, 
        FloatMatrixHostDense matrixHostDense, CGSolverFloatDevice solver)
    {
        System.out.println("Solving with "+solver.getClass().getSimpleName());
        
        // Create a representation of the input matrix in CSR
        // format that is stored on the device
        FloatMatrixDeviceCSR matrix = 
            FloatMatrices.createFloatMatrixDeviceCSR(
                cusparseHandle, matrixHostDense, 1e-8f);

        //System.out.println("Device matrix\n"+matrix);
        
        // Create the vectors for the equation matrix*x = b
        int numRows = matrix.getNumRows();
        float x[] = new float[numRows];
        float b[] = new float[numRows];
        for (int i = 0; i < numRows; i++)
        {
            b[i] = 1.0f;
            x[i] = 0.0f;
        }
        CUdeviceptr deviceX = Utils.createPointer(x);
        CUdeviceptr deviceB = Utils.createPointer(b);
        
        // Note: The time measurement here just gives a rough
        // approximation of the required time. It should NOT
        // be considered as a "benchmark"!
        long before = 0;
        long after = 0;
        
        // Setup the solver
        before = System.nanoTime();
        solver.setup(matrix);
        JCuda.cudaDeviceSynchronize();
        after = System.nanoTime();
        System.out.println("Setup duration: "+(after-before)/1e6+" ms");
        
        // Solve the system
        before = System.nanoTime();
        solver.solve(deviceX, deviceB);
        JCuda.cudaDeviceSynchronize();
        after = System.nanoTime();
        System.out.println("Solve duration: "+(after-before)/1e6+" ms");

        // Copy the result back to the host and verify it
        Utils.copyToHost(deviceX, x);
        float error = computeError(matrixHostDense, x, b);
        System.out.printf(
            "Maximum single component error in Ax-b = %e\n\n", error);
        
        // Clean up
        matrix.dispose();
        cudaFree(deviceX);
        cudaFree(deviceB);
    }
    
    
    /**
     * Create a random tridiagonal symmetric dense matrix containing
     * float values that is stored in host memory, with the given 
     * number of rows and columns.
     * 
     * @param numRows The number of rows
     * @param numCols The number of columns
     * @return The new matrix
     */
    private static FloatMatrixHostDense createMatrix(
        int numRows, int numCols)
    {
        FloatMatrixHostDense matrixHostDense = 
            FloatMatrices.createFloatMatrixHostDense(numRows, numCols);
        setRandomTridiagonalSymmetric(matrixHostDense);
        return matrixHostDense;
    }
    
    /**
     * Set the values of the given matrix (which is assumed to be
     * all zero) to be a random tridiagonal symmetric matrix.
     * 
     * @param matrix The input matrix
     */
    private static void setRandomTridiagonalSymmetric(
        MutableFloatMatrixHost matrix)
    {
        Random random = new Random(0);
        matrix.set(0,0,random.nextFloat()+30);
        for (int r=1; r<matrix.getNumRows(); r++)
        {
            matrix.set(r, r-1, random.nextFloat());
            matrix.set(r-1, r, matrix.get(r, r-1));
            matrix.set(r, r, 10 + random.nextFloat());
        }
    }

    /**
     * Compute the maximum single component error for the equation
     * <code>matrix * x = b</code>
     * 
     * @param matrix The matrix
     * @param x The solution
     * @param b The right hand side
     * @return The maximum single component error
     */
    private static float computeError(
        FloatMatrixHost matrix, float x[], float b[])
    {
        float error = -1;
        for (int r=0; r<matrix.getNumRows(); r++)
        {
            float sum = 0;
            for (int c=0; c<matrix.getNumCols(); c++)
            {
                sum += matrix.get(r, c) * x[c];
            }
            float diff = Math.abs(sum - b[r]);
            error = Math.max(error, diff);
        }
        return error;
    }
    
    
}
