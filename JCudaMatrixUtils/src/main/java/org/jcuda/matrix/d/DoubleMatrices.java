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

import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;

import java.util.Locale;

import jcuda.driver.CUdeviceptr;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;

import org.jcuda.matrix.Utils;

/**
 * Utility methods for matrix creation and conversion
 */
public class DoubleMatrices
{
    /**
     * Create a new {@link DoubleMatrixHostDense}. 
     * The result is s a dense double matrix where the data is stored in 
     * host memory, as a 1D array, in column-major order.
     * 
     * @param numRows The number of rows
     * @param numCols The number of columns
     * @return The new matrix
     */
    public static DoubleMatrixHostDense createDoubleMatrixHostDense(
        int numRows, int numCols)
    {
        DefaultDoubleMatrixHostDense result = 
            new DefaultDoubleMatrixHostDense(numRows, numCols);
        return result;
    }
    
    /**
     * Create a new {@link DoubleMatrixHostDense} that has the same 
     * contents as the given matrix. 
     * The result is s a dense double matrix where the data is stored in 
     * host memory, as a 1D array, in column-major order.
     * 
     * @param matrix The input matrix
     * @return The new matrix
     */
    public static DoubleMatrixHostDense createDoubleMatrixHostDense(
        DoubleMatrixHost matrix)
    {
        int numRows = matrix.getNumRows();
        int numCols = matrix.getNumCols();
        DefaultDoubleMatrixHostDense result = 
            new DefaultDoubleMatrixHostDense(numRows, numCols);
        for (int r=0; r<numRows; r++)
        {
            for (int c=0; c<numCols; c++)
            {
                double value = matrix.get(r, c);
                result.set(r, c, value);
            }
        }
        return result;
    }

    
    /**
     * Creates a new {@link DoubleMatrixHostCSR} with the given values. 
     * All the given values will be stored directly as a reference. 
     * 
     * @param numRows The number of rows 
     * @param numCols The number of columns
     * @param values The non-zero values
     * @param rowPointers The row pointers
     * @param columnIndices The column indices
     * @return The new matrix
     */
    public static DoubleMatrixHostCSR createDoubleMatrixHostCSR(
        int numRows, int numCols, 
        double[] values, int[] rowPointers, int[] columnIndices)
    {
        return new DefaultDoubleMatrixHostCSR(
            numRows, numCols, values.length, values, 
            rowPointers, columnIndices);
    }
    
    /**
     * Create a {@link DoubleMatrixHostCSR} that has the same contents as 
     * the given matrix. 
     * The result is a double matrix in CSR format that is stored in host 
     * memory.<br>
     * <br> 
     * Only elements whose absolute value is greater than or equal
     * to the given epsilon will be considered as being 'non-zero'. 
     * 
     * @param matrix The input matrix
     * @param epsilon The epsilon for non-zero values
     * @return The new matrix
     */
    public static DoubleMatrixHostCSR createDoubleMatrixHostCSR(
        DoubleMatrixHost matrix, double epsilon)
    {
        int numNonZeros = countNumNonZeros(matrix, epsilon);
        int numRows = matrix.getNumRows();
        int numCols = matrix.getNumCols();
        double values[] = new double[numNonZeros];
        int columnIndices[] = new int[numNonZeros];
        int rowPointers[] = new int[numRows+1];
        rowPointers[rowPointers.length-1] = numNonZeros;
        
        int index = 0;
        for (int r=0; r<numRows; r++)
        {
            boolean firstColumn = true;
            for (int c=0; c<numCols; c++)
            {
                double value = matrix.get(r, c);
                if (Math.abs(value) >= epsilon)
                {
                    if (firstColumn)
                    {
                        rowPointers[r] = index;
                        firstColumn = false;
                    }
                    values[index] = value;
                    columnIndices[index] = c;
                    index++;
                }
            }
        }
        return new DefaultDoubleMatrixHostCSR(
            numRows, numCols, numNonZeros, 
            values, rowPointers, columnIndices);
        
    }
    
    /**
     * Count the number of non-zero elements in the given matrix.
     * Only elements whose absolute value is greater than or equal
     * to the given epsilon will be counted. 
     * 
     * @param matrix The matrix
     * @param epsilon The epsilon
     * @return The number of non-zero elements
     */
    static int countNumNonZeros(DoubleMatrixHost matrix, double epsilon)
    {
        int count = 0;
        for (int r=0; r<matrix.getNumRows(); r++)
        {
            for (int c=0; c<matrix.getNumCols(); c++)
            {
                double value = matrix.get(r, c);
                if (Math.abs(value) >= epsilon)
                {
                    count++;
                }
            }
        }
        return count;
    }
    
    
    
    /**
     * Create a {@link DoubleMatrixHostDense} that has the same contents 
     * as the given dense device memory matrix. 
     * The result is s a dense double matrix where the data is stored in 
     * host memory, as a 1D array, in column-major order.
     * 
     * @param matrix The input matrix
     * @return The new matrix
     */
    public static DoubleMatrixHostDense createDoubleMatrixHostDense(
        DoubleMatrixDeviceDense matrix)
    {
        int numRows = matrix.getNumRows();
        int numCols = matrix.getNumCols();
        double[] data = 
            Utils.createDoubleArray(matrix.getData(), numRows * numCols);
        return new DefaultDoubleMatrixHostDense(numRows, numCols, data);
    }
    
    /**
     * Create a {@link DoubleMatrixHostCSR} that has the same contents as 
     * the given {@link DoubleMatrixDeviceCSR}. 
     * The result is a double matrix in CSR format that is stored in host 
     * memory.
     * 
     * @param matrix The input matrix
     * @return The new matrix
     */
    public static DoubleMatrixHostCSR createDoubleMatrixHostCSR(
        DoubleMatrixDeviceCSR matrix)
    {
        int numRows = matrix.getNumRows();
        int numCols = matrix.getNumCols();
        int numNonZeros = matrix.getNumNonZeros();
        double[] values = 
            Utils.createDoubleArray(matrix.getValues(), numNonZeros);
        int[] rowPointers = 
            Utils.createIntArray(matrix.getRowPointers(), numRows + 1);
        int[] columnIndices = 
            Utils.createIntArray(matrix.getColumnIndices(), numNonZeros);
        return new DefaultDoubleMatrixHostCSR(
            numRows, numCols, numNonZeros, 
            values, rowPointers, columnIndices);
    }
    
    /**
     * Create a {@link DoubleMatrixHostCSR} from the given parameters.
     * The result is a double matrix in CSR format that is stored in host 
     * memory, and has the specified number of rows and columns, and can 
     * store the specified number of non-zero values
     * 
     * @param numRows The number of rows
     * @param numCols The number of columns
     * @param numNonZeros The number of non-zero elements
     * @return The new matrix
     */
    public static DoubleMatrixHostCSR createDoubleMatrixHostCSR(
        int numRows, int numCols, int numNonZeros)
    {
        return new DefaultDoubleMatrixHostCSR(numRows, numCols, numNonZeros);
    }
    
    
    /**
     * Create a {@link DoubleMatrixDeviceCSR}. The result is a double matrix
     * in CSR format that is stored in device memory and has the same 
     * contents as the given matrix that is stored in host memory.
     * Only elements whose absolute value is greater than or equal
     * to the given epsilon will be considered as being 'non-zero'. 
     * 
     * @param handle The CUSPARSE context handle
     * @param matrix The input matrix.
     * @param epsilon The epsilon for non-zero values
     * @return The new matrix
     */
    public static DoubleMatrixDeviceCSR createDoubleMatrixDeviceCSR(
        cusparseHandle handle, DoubleMatrixHostDense matrix, double epsilon)
    {
        return createDoubleMatrixDeviceCSR(handle, 
            createDoubleMatrixHostCSR(matrix, epsilon));
    }
    
    /**
     * Create a {@link DoubleMatrixDeviceCSR}. The result is a double matrix 
     * in CSR format that is stored in device memory and has the same 
     * contents as the given matrix that is stored in host memory.
     * 
     * @param handle The CUSPARSE context handle
     * @param matrix The input matrix.
     * @return The new matrix
     */
    public static DoubleMatrixDeviceCSR createDoubleMatrixDeviceCSR(
        cusparseHandle handle, DoubleMatrixHostCSR matrix)
    {
        cusparseMatDescr descr = new cusparseMatDescr();
        cusparseCreateMatDescr(descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        int numRows = matrix.getNumRows();
        int numCols = matrix.getNumCols();
        int numNonZeros = matrix.getNumNonZeros();
        
        CUdeviceptr values = 
            Utils.createPointer(matrix.getValues());
        CUdeviceptr rowPointers = 
            Utils.createPointer(matrix.getRowPointers());
        CUdeviceptr columnIndices = 
            Utils.createPointer(matrix.getColumnIndices());

        return new DefaultDoubleMatrixDeviceCSR(
            descr, numRows, numCols, numNonZeros, 
            values, rowPointers, columnIndices);
    }
    
    
    /**
     * Creates a new {@link DoubleMatrixDeviceCSR} with the given values. 
     * All the given values will be stored directly. Thus, freeing the given
     * pointers will be done when the {@link DoubleMatrixDeviceCSR#dispose()} 
     * method of the returned matrix is called.
     * 
     * @param descr The CUSPARSE matrix descriptor
     * @param numRows The number of rows 
     * @param numCols The number of columns
     * @param numNonZeros The number of non-zero elements 
     * @param values The non-zero values
     * @param rowPointers The row pointers
     * @param columnIndices The column indices
     * @return The new matrix
     */
    public static DoubleMatrixDeviceCSR createDoubleMatrixDeviceCSR(
        cusparseMatDescr descr, int numRows, int numCols, int numNonZeros, 
        CUdeviceptr values, CUdeviceptr rowPointers, CUdeviceptr columnIndices)
    {
        return new DefaultDoubleMatrixDeviceCSR(descr, 
            numRows, numCols, numNonZeros, 
            values, rowPointers, 
            columnIndices);
    }
    
    
    

    
    /**
     * Creates an unspecified String representation of the given matrix. 
     * If the matrix is too large, only the upper left part will be 
     * printed.    
     *
     * @param matrix The matrix
     * @return The String representation
     */
    public static String toString2D(DoubleMatrixHost matrix)
    {
        return toString2D(matrix, 16);
    }

    /**
     * Creates a String representation of the given matrix, using
     * the default locale and an unspecified format for the entries.
     * If the matrix is larger than the specified maximum size, 
     * then only the upper left part will be printed.    
     *
     * @param matrix The matrix
     * @param maxSize The maximum size of the printed matrix
     * @return The String representation
     */
    public static String toString2D(DoubleMatrixHost matrix, int maxSize)
    {
        return toString2D(matrix, maxSize, Locale.getDefault(), "%8.3f "); 
    }
    
    /**
     * Creates a String representation of the given matrix. 
     * If the matrix is larger than the specified maximum size, 
     * then only the upper left part will be printed.    
     *
     * @param matrix The matrix
     * @param maxSize The maximum size of the printed matrix
     * @param locale The locale
     * @param format The format string for each entry
     * @return The String representation
     */
    public static String toString2D(
        DoubleMatrixHost matrix, int maxSize, Locale locale, String format)
    {
        StringBuilder sb = new StringBuilder();
        for (int r=0; r<matrix.getNumRows(); r++)
        {
            for (int c=0; c<matrix.getNumCols(); c++)
            {
                sb.append(String.format(locale, 
                    format, matrix.get(r,c)));
                
                if (c == maxSize)
                {
                    sb.append("...");
                    break;
                }
            }
            sb.append("\n");
            if (r == maxSize)
            {
                sb.append("...");
                break;
            }
        }
        return sb.toString();
    }
    
    
    /**
     * Private constructor to prevent instantiation
     */
    private DoubleMatrices()
    {
        // Private constructor to prevent instantiation   
    }

    
}
