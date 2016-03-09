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

import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseSetMatDiagType;
import static jcuda.jcusparse.JCusparse.cusparseSetMatFillMode;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_LOWER;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_TRIANGULAR;
import jcuda.driver.CUdeviceptr;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;

import org.jcuda.matrix.Utils;
import org.jcuda.matrix.f.FloatMatrices;
import org.jcuda.matrix.f.FloatMatrixDeviceCSR;
import org.jcuda.matrix.f.FloatMatrixHostCSR;

/**
 * Utility methods for the CG solver classes
 */
class CGUtilsFloat
{
    /**
     * Generate the incomplete Cholesky factor H (lower triangular) for 
     * the given matrix, in CSR format, stored in device memory. <br>
     * <br>
     * <strong>NOTE:</strong> This method is rather inefficient, because 
     * it copies the data from the device to the host, creates the result 
     * matrix in host memory, and copies the data back from the host to 
     * the device. This could be optimized by using an own kernel for
     * the creation of the ICP matrix.
     *  
     * @param cusparseHandle The CUSPARSE context handle
     * @param matrix The input matrix
     * @return The new matrix
     */
    static FloatMatrixDeviceCSR createDeviceICP(
        cusparseHandle cusparseHandle, FloatMatrixDeviceCSR matrix)
    {
        FloatMatrixHostCSR matrixHostCSR = 
            FloatMatrices.createFloatMatrixHostCSR(matrix);
        int row[] = matrixHostCSR.getRowPointers();
        float val[] = matrixHostCSR.getValues();
        
        int numRows = matrix.getNumRows();
        int numNonZerosICP = 2 * numRows - 1;
        FloatMatrixHostCSR hostICP = 
            FloatMatrices.createFloatMatrixHostCSR(
                numRows, numRows, numNonZerosICP);
        genICP(row, val, numRows, hostICP.getColumnIndices(), 
            hostICP.getRowPointers(), hostICP.getValues());
        
        cusparseMatDescr descr = new cusparseMatDescr();
        cusparseCreateMatDescr(descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
        cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
        
        CUdeviceptr values = 
            Utils.createPointer(hostICP.getValues());
        CUdeviceptr rowPointers = 
            Utils.createPointer(hostICP.getRowPointers());
        CUdeviceptr columnIndices = 
            Utils.createPointer(hostICP.getColumnIndices());
        
        return FloatMatrices.createFloatMatrixDeviceCSR(
            descr, numRows, numRows, numNonZerosICP, 
            values, rowPointers, columnIndices);
    }
    
    /**
     * Generate the Incomplete Cholesky factor H (lower triangular).
     * 
     * Adapted from the NVIDIA conjugate gradient sample.
     * 
     * @param rowPointers The row pointers
     * @param values The values
     * @param numRows The number of rows
     * @param columnIndicesICP The column indices of the ICP
     * @param rowPointersICP The row pointers of the ICP
     * @param valuesICP The value of the ICP
     */
    private static void genICP(int rowPointers[], float values[], int numRows, 
        int columnIndicesICP[], int rowPointersICP[], float valuesICP[])
    {
        rowPointersICP[0] = 0;
        columnIndicesICP[0] = 0;
        int inz = 1;

        for (int k = 1; k < numRows; k++)
        {
            rowPointersICP[k] = inz;
            for (int j = k - 1; j <= k; j++)
            {
                columnIndicesICP[inz] = j;
                inz++;
            }
        }
        rowPointersICP[numRows] = inz;

        valuesICP[0] = values[0];
        for (int k = 1; k < numRows; k++)
        {
            int rp = rowPointers[k];
            int rpICP = rowPointersICP[k];
            valuesICP[rpICP] = values[rp];
            valuesICP[rpICP + 1] = values[rp + 1];
        }

        for (int k = 0; k < numRows; k++)
        {
            int rpICP = rowPointersICP[k + 1];
            valuesICP[rpICP - 1] = (float)Math.sqrt(valuesICP[rpICP - 1]);
            if (k < numRows - 1)
            {
                valuesICP[rpICP] /= valuesICP[rpICP - 1];
                valuesICP[rpICP + 1] -= valuesICP[rpICP] * valuesICP[rpICP];
            }
        }
    }
    
    /**
     * Private constructor to prevent instantiation
     */
    private CGUtilsFloat()
    {
        // Private constructor to prevent instantiation
    }

}
