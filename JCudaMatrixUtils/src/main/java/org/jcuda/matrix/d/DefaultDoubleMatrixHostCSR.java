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


/**
 * Default implementation of a {@link DoubleMatrixHostCSR}
 */
class DefaultDoubleMatrixHostCSR implements DoubleMatrixHostCSR
{
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
    private final double values[];
    
    /**
     * The row pointers 
     */
    private final int rowPointers[];
    
    /**
     * The column indices 
     */
    private final int columnIndices[];
    
    /**
     * Creates a new matrix with the given values. 
     * 
     * @param numRows The number of rows 
     * @param numCols The number of columns
     * @param numNonZeros The number of non-zero elements 
     * @param values The non-zero values
     * @param rowPointers The row pointers
     * @param columnIndices The column indices
     */
    DefaultDoubleMatrixHostCSR(int numRows, int numCols, int numNonZeros)
    {
        this.numRows = numRows;
        this.numCols = numCols;
        this.numNonZeros = numNonZeros;
        this.values = new double[numNonZeros];
        this.rowPointers = new int[numRows+1];
        this.columnIndices = new int[numNonZeros];
    }
    
    /**
     * Creates a new matrix with the given values. All the given
     * values will be stored directly as a reference. 
     * 
     * @param numRows The number of rows 
     * @param numCols The number of columns
     * @param numNonZeros The number of non-zero elements 
     * @param values The non-zero values
     * @param rowPointers The row pointers
     * @param columnIndices The column indices
     */
    DefaultDoubleMatrixHostCSR(
        int numRows, int numCols, int numNonZeros, 
        double[] values, int[] rowPointers, int[] columnIndices)
    {
        this.numRows = numRows;
        this.numCols = numCols;
        this.numNonZeros = numNonZeros;
        this.values = values;
        this.rowPointers = rowPointers;
        this.columnIndices = columnIndices;
    }
    
    @Override
    public int getNumNonZeros()
    {
        return numNonZeros;
    }

    @Override
    public double[] getValues()
    {
        return values;
    }

    @Override
    public int[] getRowPointers()
    {
        return rowPointers;
    }

    @Override
    public int[] getColumnIndices()
    {
        return columnIndices;
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
    public double get(int r, int c)
    {
        int c0 = rowPointers[r];
        int c1 = rowPointers[r+1];
        for (int cc=c0; cc<c1; cc++)
        {
            if (columnIndices[cc] == c)
            {
                return values[cc];
            }
        }
        return 0;
    }
    
    
    @Override
    public String toString()
    {
        return DoubleMatrices.toString2D(this);
    }
}