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

package org.jcuda.matrix.d;

import static jcuda.jcusparse.JCusparse.cusparseDestroyMatDescr;
import static jcuda.runtime.JCuda.cudaFree;
import jcuda.driver.CUdeviceptr;
import jcuda.jcusparse.cusparseMatDescr;

import org.jcuda.matrix.f.FloatMatrixDeviceCSR;

/**
 * Default implementation of a {@link FloatMatrixDeviceCSR}
 */
class DefaultDoubleMatrixDeviceCSR implements DoubleMatrixDeviceCSR
{
    /**
     * The CUSPARSE matrix descriptor
     */
    private final cusparseMatDescr descr;
    
    /**
     * The number of rows 
     */
    private final int numRows;
    
    /**
     * The number of columns 
     */
    private final int numCols;

    /**
     * The number of non-zero elements in this matrix
     */
    private final int numNonZeros;
    
    /**
     * The non-zero values 
     */
    private final CUdeviceptr values;
    
    /**
     * The row pointers
     */
    private final CUdeviceptr rowPointers;
    
    /**
     * The column indices 
     */
    private final CUdeviceptr columnIndices;
    
    /**
     * Creates a new matrix with the given values. All the given
     * values will be stored directly. Thus, freeing the given
     * pointers will be done in the {@link #dispose()} method
     * of this class.
     * 
     * @param descr The CUSPARSE matrix descriptor
     * @param numRows The number of rows 
     * @param numCols The number of columns
     * @param numNonZeros The number of non-zero elements 
     * @param values The non-zero values
     * @param rowPointers The row pointers
     * @param columnIndices The column indices
     */
    DefaultDoubleMatrixDeviceCSR(cusparseMatDescr descr, 
        int numRows, int numCols, int numNonZeros, 
        CUdeviceptr values, CUdeviceptr rowPointers, 
        CUdeviceptr columnIndices)
    {
        this.descr = descr;
        this.numRows = numRows;
        this.numCols = numCols;
        this.numNonZeros = numNonZeros;
        this.values = values;
        this.rowPointers = rowPointers;
        this.columnIndices = columnIndices;
    }
    
    @Override
    public void dispose()
    {
        cudaFree(values);
        cudaFree(rowPointers);
        cudaFree(columnIndices);
        cusparseDestroyMatDescr(descr);
    }
    
    @Override
    public int getNumRows()
    {
        return numRows;
    }

    @Override
    public int getNumCols()
    {
        return numCols;
    }
    
    @Override
    public int getNumNonZeros()
    {
        return numNonZeros;
    }

    @Override
    public CUdeviceptr getValues()
    {
        return values;
    }

    @Override
    public CUdeviceptr getRowPointers()
    {
        return rowPointers;
    }

    @Override
    public CUdeviceptr getColumnIndices()
    {
        return columnIndices;
    }
    
    @Override
    public cusparseMatDescr getDescriptor()
    {
        return descr;
    }
    

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("Copied from device:\n");
        sb.append(DoubleMatrices.createDoubleMatrixHostCSR(this));
        return sb.toString();
    }


}