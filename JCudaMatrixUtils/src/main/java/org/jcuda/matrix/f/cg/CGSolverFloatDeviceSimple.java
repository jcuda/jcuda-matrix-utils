/*
 * JCudaMatrixUtils - Matrix utility classes for JCuda
 *
 * Copyright (c) 2010-2016 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

package org.jcuda.matrix.f.cg;

import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcusparse.JCusparse.cusparseScsrmv;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.runtime.JCuda.*;

import org.jcuda.matrix.f.FloatMatrixDeviceCSR;

import jcuda.*;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.*;

/**
 * Simple implementation of the {@link CGSolverFloatDevice} interface
 */
class CGSolverFloatDeviceSimple implements CGSolverFloatDevice
{
    /**
     * The handle to the CUSPARSE context
     */
    private final cusparseHandle cusparseHandle;
    
    /**
     * The handle to the CUBLAS context
     */
    private final cublasHandle cublasHandle;
    
    /**
     * The matrix used by this solver
     */
    private FloatMatrixDeviceCSR matrix;
    
    /**
     * The tolerance
     */
    private float tolerance;
    
    /**
     * The maximum number of iterations
     */
    private int maxIterations;
    
    /**
     * Temporary variables owned by this solver.
     */
    private Pointer p;

    /**
     * Temporary variables owned by this solver.
     */
    private Pointer omega;
    
    /**
     * Create a new solver using the given CUSPARSE and CUBLAS handles. 
     * The tolerance will be 1e-8f, and the maximum number of 
     * iterations will be 100.
     * 
     * @param cusparseHandle The CUSPARSE handle
     * @param cublasHandle The CUBLAS handle
     */
    CGSolverFloatDeviceSimple(
        cusparseHandle cusparseHandle, 
        cublasHandle cublasHandle)
    {
        this.cusparseHandle = cusparseHandle;
        this.cublasHandle = cublasHandle;
        
        this.tolerance = 1e-8f;
        this.maxIterations = 100;
        
    }
    
    @Override
    public void dispose()
    {
        if (matrix != null)
        {
            cudaFree(p);
            cudaFree(omega);
        }
    }
    
    @Override
    public void setTolerance(float tolerance)
    {
        this.tolerance = tolerance;
    }

    @Override
    public float getTolerance()
    {
        return tolerance;
    }

    @Override
    public void setMaxIterations(int maxIterations)
    {
        this.maxIterations = maxIterations;
    }

    @Override
    public int getMaxIterations()
    {
        return maxIterations;
    }

    @Override
    public void setup(FloatMatrixDeviceCSR matrix)
    {
        if (this.matrix != null)
        {
            dispose();
        }
        
        this.matrix = matrix;
        int numRows = matrix.getNumRows();
        
        this.p = new Pointer();
        cudaMalloc(p, numRows * Sizeof.FLOAT);
        
        this.omega = new Pointer();
        cudaMalloc(omega, numRows * Sizeof.FLOAT);
    }
    
    @Override
    public void solve(Pointer x, Pointer b)
    {
        System.out.printf("Convergence of conjugate gradient without preconditioning: \n");

        int numRows = matrix.getNumRows();
        cusparseMatDescr descr = matrix.getDescriptor();
        
        Pointer one = Pointer.to(new float[]{1.0f});
        Pointer zero = Pointer.to(new float[]{0.0f});
        
        float resultArray[] = new float[1];
        Pointer resultPointer = Pointer.to(resultArray);

        float alphaArray[] = new float[1];
        Pointer alphaPointer = Pointer.to(alphaArray);

        float betaArray[] = new float[1];
        Pointer betaPointer = Pointer.to(betaArray);
        
        
        int k = 0;
        float r0 = 0;
        float r1 = 0;
        cublasSdot(cublasHandle, numRows, b, 1, b, 1, resultPointer);
        r1 = resultArray[0];
        while (r1 > tolerance * tolerance && k <= maxIterations)
        {
            k++;
            if (k == 1)
            {
                cublasScopy(cublasHandle, numRows, b, 1, p, 1);
            }
            else
            {
                betaArray[0] = r1 / r0;
                cublasSscal(cublasHandle, numRows, betaPointer, p, 1);
                
                cublasSaxpy(cublasHandle, numRows, one, b, 1, p, 1);
            }
            cusparseScsrmv(cusparseHandle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numRows, 
                matrix.getNumNonZeros(), one, descr, matrix.getValues(), 
                matrix.getRowPointers(), matrix.getColumnIndices(), 
                p, zero, omega);
            cublasSdot(cublasHandle, numRows, p, 1, omega, 1, resultPointer);
            alphaArray[0] = r1 / resultArray[0];
            cublasSaxpy(cublasHandle, numRows, alphaPointer, p, 1, x, 1);
            alphaArray[0] = -alphaArray[0];
            cublasSaxpy(cublasHandle, numRows, alphaPointer, omega, 1, b, 1);
            r0 = r1;
            cublasSdot(cublasHandle, numRows, b, 1, b, 1, resultPointer);
            r1 = resultArray[0];
            System.out.printf("  iteration = %3d, residual = %e \n", k, Math.sqrt(r1));
        }
    }
}