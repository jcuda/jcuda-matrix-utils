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

package org.jcuda.matrix.f;


/**
 * Default implementation of a {@link FloatMatrixHostDense}.
 * The data is stored in column-major order.
 */
class DefaultFloatMatrixHostDense implements FloatMatrixHostDense
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
     * The data 
     */
    private final float data[];
    
    /**
     * Creates a new matrix with the given number of rows and columns
     * 
     * @param numRows The number of rows
     * @param numCols The number of columns
     */
    DefaultFloatMatrixHostDense(int numRows, int numCols)
    {
        this(numRows, numCols, new float[numRows * numCols]);
    }
    
    /**
     * Creates a new matrix with the given number of rows and columns,
     * and the given data
     * 
     * @param numRows The number of rows
     * @param numCols The number of columns
     * @param data The data 
     */
    DefaultFloatMatrixHostDense(int numRows, int numCols, float data[])
    {
        this.numRows = numRows;
        this.numCols = numCols;
        this.data = data;
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
    public float get(int r, int c)
    {
        return data[c+r*getNumCols()];
    }
    @Override
    public float[] getData()
    {
        return data;
    }
    
    @Override
    public void set(int r, int c, float value)
    {
        data[c+r*getNumCols()] = value;
    }
    
    @Override
    public String toString()
    {
        return FloatMatrices.toString2D(this);
    }
    
}