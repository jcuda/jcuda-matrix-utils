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

package org.jcuda.matrix.d.cg;

import static jcuda.jcublas.JCublas2.cublasDaxpy;
import static jcuda.jcublas.JCublas2.cublasDcopy;
import static jcuda.jcublas.JCublas2.cublasDdot;
import static jcuda.jcublas.JCublas2.cublasDscal;
import static jcuda.jcusparse.JCusparse.cusparseCreateSolveAnalysisInfo;
import static jcuda.jcusparse.JCusparse.cusparseDcsrmv;
import static jcuda.jcusparse.JCusparse.cusparseDcsrsv_analysis;
import static jcuda.jcusparse.JCusparse.cusparseDcsrsv_solve;
import static jcuda.jcusparse.JCusparse.cusparseDestroySolveAnalysisInfo;
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

import org.jcuda.matrix.d.DoubleMatrixDeviceCSR;

/**
 * Implementation of the {@link CGSolverDoubleDevice} interface that solves the
 * equation using an incomplete Cholesky preconditioner
 */
class CGSolverDoubleDevicePrecond implements CGSolverDoubleDevice
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
    private DoubleMatrixDeviceCSR matrix;

    /**
     * The preconditioner matrix
     */
    private DoubleMatrixDeviceCSR deviceICP;
    
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
    private double tolerance;
    
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
    CGSolverDoubleDevicePrecond(
        cusparseHandle cusparseHandle, 
        cublasHandle cublasHandle)
    {
        this.cusparseHandle = cusparseHandle;
        this.cublasHandle = cublasHandle;

        this.tolerance = 1e-16f;
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
    public void setTolerance(double tolerance)
    {
        this.tolerance = tolerance;
    }

    @Override
    public double getTolerance()
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
    public void setup(DoubleMatrixDeviceCSR matrix)
    {
        if (this.matrix != null)
        {
            dispose();
        }
        
        this.matrix = matrix;
        int numRows = matrix.getNumRows();
        
        this.p = new Pointer();
        cudaMalloc(p, numRows * Sizeof.DOUBLE);
        
        this.omega = new Pointer();
        cudaMalloc(omega, numRows * Sizeof.DOUBLE);

        this.y = new Pointer();
        cudaMalloc(y, numRows * Sizeof.DOUBLE);
        
        this.zm1 = new Pointer();
        cudaMalloc(zm1, numRows * Sizeof.DOUBLE);
        
        this.zm2 = new Pointer();
        cudaMalloc(zm2, numRows * Sizeof.DOUBLE);
        
        this.rm2 = new Pointer();
        cudaMalloc(rm2, numRows * Sizeof.DOUBLE);
        
        // Create the incomplete Cholesky Preconditioner
        deviceICP =  CGUtilsDouble.createDeviceICP(cusparseHandle, matrix);
        int numNonZerosICP = deviceICP.getNumNonZeros();
        cusparseMatDescr descrM = deviceICP.getDescriptor(); 
        
        // Create the analysis info object for the Non-Transpose case
        info = new cusparseSolveAnalysisInfo();
        cusparseCreateSolveAnalysisInfo(info);
        cusparseDcsrsv_analysis(cusparseHandle, 
            CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numNonZerosICP, 
            descrM, deviceICP.getValues(), deviceICP.getRowPointers(),
            deviceICP.getColumnIndices(), info);

        // Create the analysis info object for the Transpose case
        infoTrans = new cusparseSolveAnalysisInfo();
        cusparseCreateSolveAnalysisInfo(infoTrans);
        cusparseDcsrsv_analysis(cusparseHandle, 
            CUSPARSE_OPERATION_TRANSPOSE, numRows, numNonZerosICP, 
            descrM, deviceICP.getValues(), deviceICP.getRowPointers(), 
            deviceICP.getColumnIndices(), infoTrans);
        
    }
    
    @Override
    public void solve(Pointer x, Pointer b)
    {
        System.out.printf("Convergence of conjugate gradient using Incomplete Cholesky preconditioning: \n");

        Pointer one = Pointer.to(new double[]{1.0f});
        Pointer zero = Pointer.to(new double[]{0.0f});
        
        double resultArray[] = new double[1];
        Pointer resultPointer = Pointer.to(resultArray);

        double resultArray0[] = new double[1];
        Pointer resultPointer0 = Pointer.to(resultArray0);

        double resultArray1[] = new double[1];
        Pointer resultPointer1 = Pointer.to(resultArray1);
        
        double alphaArray[] = new double[1];
        Pointer alphaPointer = Pointer.to(alphaArray);

        double betaArray[] = new double[1];
        Pointer betaPointer = Pointer.to(betaArray);
        
        int numRows = matrix.getNumRows();
        cusparseMatDescr descr = matrix.getDescriptor();
        cusparseMatDescr descrM = deviceICP.getDescriptor();
        
        // Preconditioned Conjugate Gradient. Follows the description by Golub &
        // Van Loan, "Matrix Computations 3rd ed.", Algorithm 10.3.1
        int k = 0;
        double r1 = 0;
        cublasDdot(cublasHandle, numRows, b, 1, b, 1, resultPointer);
        r1 = resultArray[0];
        while (r1 > tolerance * tolerance && k <= maxIterations)
        {
            // Solve M z = H H^T z = r
            // Forward Solve: H y = r
            cusparseDcsrsv_solve(cusparseHandle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, one, 
                descrM, deviceICP.getValues(), deviceICP.getRowPointers(), 
                deviceICP.getColumnIndices(), info, b, y);

            // Back Substitution: H^T z = y
            cusparseDcsrsv_solve(cusparseHandle, 
                CUSPARSE_OPERATION_TRANSPOSE, numRows, one, 
                descrM, deviceICP.getValues(), deviceICP.getRowPointers(), 
                deviceICP.getColumnIndices(), infoTrans, y, zm1);

            k++;
            if (k == 1)
            {
                cublasDcopy(cublasHandle, numRows, zm1, 1, p, 1);
            }
            else
            {
                cublasDdot(cublasHandle, 
                    numRows, b, 1, zm1, 1, resultPointer0);
                cublasDdot(cublasHandle, 
                    numRows, rm2, 1, zm2, 1, resultPointer1);
                betaArray[0] = resultArray0[0] / resultArray1[0];
                cublasDscal(cublasHandle, numRows, betaPointer, p, 1);
                cublasDaxpy(cublasHandle, numRows, one, zm1, 1, p, 1);
            }
            cusparseDcsrmv(cusparseHandle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numRows, 
                matrix.getNumNonZeros(), one, descr, matrix.getValues(), 
                matrix.getRowPointers(), matrix.getColumnIndices(), 
                p, zero, omega);
            
            cublasDdot(cublasHandle, numRows, b, 1, zm1, 1, resultPointer0);
            cublasDdot(cublasHandle, numRows, p, 1, omega, 1, resultPointer1);
            alphaArray[0] = resultArray0[0] / resultArray1[0];
            cublasDaxpy(cublasHandle, numRows, alphaPointer, p, 1, x, 1);
            cublasDcopy(cublasHandle, numRows, b, 1, rm2, 1);
            cublasDcopy(cublasHandle, numRows, zm1, 1, zm2, 1);
            alphaArray[0] = -alphaArray[0];
            cublasDaxpy(cublasHandle, numRows, alphaPointer, omega, 1, b, 1);
            cublasDdot(cublasHandle, numRows, b, 1, b, 1, resultPointer);
            r1 = resultArray[0];
            System.out.printf("  iteration = %3d, residual = %e \n", k, Math.sqrt(r1));
        }
    }
}
