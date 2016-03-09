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

import static jcuda.jcublas.JCublas2.cublasSaxpy;
import static jcuda.jcublas.JCublas2.cublasScopy;
import static jcuda.jcublas.JCublas2.cublasSdot;
import static jcuda.jcublas.JCublas2.cublasSscal;
import static jcuda.jcusparse.JCusparse.cusparseCreateSolveAnalysisInfo;
import static jcuda.jcusparse.JCusparse.cusparseDestroySolveAnalysisInfo;
import static jcuda.jcusparse.JCusparse.cusparseScsrmv;
import static jcuda.jcusparse.JCusparse.cusparseScsrsv_analysis;
import static jcuda.jcusparse.JCusparse.cusparseScsrsv_solve;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparseSolveAnalysisInfo;

import org.jcuda.matrix.f.FloatMatrixDeviceCSR;

/**
 * Implementation of the {@link CGSolverFloatDevice} interface that solves the
 * equation using an incomplete Cholesky preconditioner
 */
class CGSolverFloatDevicePrecond implements CGSolverFloatDevice
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
     * The preconditioner matrix
     */
    private FloatMatrixDeviceCSR deviceICP;
    
    /**
     * The CUSPARSE analysis info
     */
    private cusparseSolveAnalysisInfo info;
    
    /**
     * The CUSPARSE analysis info for the transposed matrix
     */
    private cusparseSolveAnalysisInfo infoTrans;
    
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
     * Temporary variables owned by this solver. 
     */
    private Pointer y;

    /**
     * Temporary variables owned by this solver. 
     */
    private Pointer zm1;

    /**
     * Temporary variables owned by this solver. 
     */
    private Pointer zm2;

    /**
     * Temporary variables owned by this solver. 
     */
    private Pointer rm2;
    
    
    /**
     * Create a new solver using the given CUSPARSE and CUBLAS handles. 
     * The tolerance will be 1e-8f, and the maximum number of 
     * iterations will be 100.
     * 
     * @param cusparseHandle The CUSPARSE handle
     * @param cublasHandle The CUBLAS handle
     */
    CGSolverFloatDevicePrecond(
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
            cudaFree(y);
            cudaFree(zm1);
            cudaFree(zm2);
            cudaFree(rm2);
            
            deviceICP.dispose();
            
            cusparseDestroySolveAnalysisInfo(info);
            cusparseDestroySolveAnalysisInfo(infoTrans);
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

        this.y = new Pointer();
        cudaMalloc(y, numRows * Sizeof.FLOAT);
        
        this.zm1 = new Pointer();
        cudaMalloc(zm1, numRows * Sizeof.FLOAT);
        
        this.zm2 = new Pointer();
        cudaMalloc(zm2, numRows * Sizeof.FLOAT);
        
        this.rm2 = new Pointer();
        cudaMalloc(rm2, numRows * Sizeof.FLOAT);
        
        // Create the incomplete Cholesky Preconditioner
        deviceICP =  CGUtilsFloat.createDeviceICP(cusparseHandle, matrix);
        int numNonZerosICP = deviceICP.getNumNonZeros();
        cusparseMatDescr descrM = deviceICP.getDescriptor(); 
        
        // Create the analysis info object for the Non-Transpose case
        info = new cusparseSolveAnalysisInfo();
        cusparseCreateSolveAnalysisInfo(info);
        cusparseScsrsv_analysis(cusparseHandle, 
            CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numNonZerosICP, 
            descrM, deviceICP.getValues(), deviceICP.getRowPointers(),
            deviceICP.getColumnIndices(), info);

        // Create the analysis info object for the Transpose case
        infoTrans = new cusparseSolveAnalysisInfo();
        cusparseCreateSolveAnalysisInfo(infoTrans);
        cusparseScsrsv_analysis(cusparseHandle, 
            CUSPARSE_OPERATION_TRANSPOSE, numRows, numNonZerosICP, 
            descrM, deviceICP.getValues(), deviceICP.getRowPointers(), 
            deviceICP.getColumnIndices(), infoTrans);
        
    }
    
    @Override
    public void solve(Pointer x, Pointer b)
    {
        System.out.printf("Convergence of conjugate gradient using Incomplete Cholesky preconditioning: \n");

        Pointer one = Pointer.to(new float[]{1.0f});
        Pointer zero = Pointer.to(new float[]{0.0f});
        
        float resultArray[] = new float[1];
        Pointer resultPointer = Pointer.to(resultArray);

        float resultArray0[] = new float[1];
        Pointer resultPointer0 = Pointer.to(resultArray0);

        float resultArray1[] = new float[1];
        Pointer resultPointer1 = Pointer.to(resultArray1);
        
        float alphaArray[] = new float[1];
        Pointer alphaPointer = Pointer.to(alphaArray);

        float betaArray[] = new float[1];
        Pointer betaPointer = Pointer.to(betaArray);
        
        int numRows = matrix.getNumRows();
        cusparseMatDescr descr = matrix.getDescriptor();
        cusparseMatDescr descrM = deviceICP.getDescriptor();
        
        // Preconditioned Conjugate Gradient. Follows the description by Golub &
        // Van Loan, "Matrix Computations 3rd ed.", Algorithm 10.3.1
        int k = 0;
        float r1 = 0;
        cublasSdot(cublasHandle, numRows, b, 1, b, 1, resultPointer);
        r1 = resultArray[0];
        while (r1 > tolerance * tolerance && k <= maxIterations)
        {
            // Solve M z = H H^T z = r
            // Forward Solve: H y = r
            cusparseScsrsv_solve(cusparseHandle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, one, 
                descrM, deviceICP.getValues(), deviceICP.getRowPointers(), 
                deviceICP.getColumnIndices(), info, b, y);

            // Back Substitution: H^T z = y
            cusparseScsrsv_solve(cusparseHandle, 
                CUSPARSE_OPERATION_TRANSPOSE, numRows, one, 
                descrM, deviceICP.getValues(), deviceICP.getRowPointers(), 
                deviceICP.getColumnIndices(), infoTrans, y, zm1);

            k++;
            if (k == 1)
            {
                cublasScopy(cublasHandle, numRows, zm1, 1, p, 1);
            }
            else
            {
                cublasSdot(cublasHandle, 
                    numRows, b, 1, zm1, 1, resultPointer0);
                cublasSdot(cublasHandle, 
                    numRows, rm2, 1, zm2, 1, resultPointer1);
                betaArray[0] = resultArray0[0] / resultArray1[0];
                cublasSscal(cublasHandle, numRows, betaPointer, p, 1);
                cublasSaxpy(cublasHandle, numRows, one, zm1, 1, p, 1);
            }
            cusparseScsrmv(cusparseHandle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numRows, 
                matrix.getNumNonZeros(), one, descr, matrix.getValues(), 
                matrix.getRowPointers(), matrix.getColumnIndices(), 
                p, zero, omega);
            
            cublasSdot(cublasHandle, numRows, b, 1, zm1, 1, resultPointer0);
            cublasSdot(cublasHandle, numRows, p, 1, omega, 1, resultPointer1);
            alphaArray[0] = resultArray0[0] / resultArray1[0];
            cublasSaxpy(cublasHandle, numRows, alphaPointer, p, 1, x, 1);
            cublasScopy(cublasHandle, numRows, b, 1, rm2, 1);
            cublasScopy(cublasHandle, numRows, zm1, 1, zm2, 1);
            alphaArray[0] = -alphaArray[0];
            cublasSaxpy(cublasHandle, numRows, alphaPointer, omega, 1, b, 1);
            cublasSdot(cublasHandle, numRows, b, 1, b, 1, resultPointer);
            r1 = resultArray[0];
            System.out.printf("  iteration = %3d, residual = %e \n", k, Math.sqrt(r1));
        }
    }
}
