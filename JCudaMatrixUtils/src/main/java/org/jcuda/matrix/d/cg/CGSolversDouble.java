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

import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.cusparseHandle;

/**
 * Methods to create {@link CGSolverDoubleDevice} instances
 */
public class CGSolversDouble
{
    /**
     * Create a simple (non-preconditioned) {@link CGSolverDoubleDevice}
     * 
     * @param cusparseHandle The CUSPARSE handle
     * @param cublasHandle The CUBLAS handle
     * @return The {@link CGSolverDoubleDevice}
     */
    public static CGSolverDoubleDevice createSimple(
        cusparseHandle cusparseHandle, cublasHandle cublasHandle)
    {
        return new CGSolverDoubleDeviceSimple(cusparseHandle, cublasHandle);
    }

    /**
     * Create a preconditioned {@link CGSolverDoubleDevice}
     * 
     * @param cusparseHandle The CUSPARSE handle
     * @param cublasHandle The CUBLAS handle
     * @return The {@link CGSolverDoubleDevice}
     */
    public static CGSolverDoubleDevice createPrecond(
        cusparseHandle cusparseHandle, cublasHandle cublasHandle)
    {
        return new CGSolverDoubleDevicePrecond(cusparseHandle, cublasHandle);
    }
    
    
    /**
     * Private constructor to prevent instantiation
     */
    private CGSolversDouble()
    {
        // Private constructor to prevent instantiation
    }
}
