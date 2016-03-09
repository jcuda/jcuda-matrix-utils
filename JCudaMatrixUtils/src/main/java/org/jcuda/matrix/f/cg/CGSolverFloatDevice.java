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

import org.jcuda.matrix.DeviceData;
import org.jcuda.matrix.f.FloatMatrixDeviceCSR;

import jcuda.Pointer;

/**
 * Interface for a CG solver that operates on matrices in CSR
 * format that are stored on the device
 */
public interface CGSolverFloatDevice extends CGSolverFloat, DeviceData
{
    /**
     * Prepare this solver to solve an equation involving the
     * given matrix. Any data structures that may already have
     * been allocated by a previous call to this method will
     * be released by calling the {@link #dispose()} method,
     * if necessary.
     * 
     * @param matrix The matrix
     */
    void setup(FloatMatrixDeviceCSR matrix);
    
    /**
     * Solve the equation <code>matrix * x = b</code> for <code>x</code>,
     * using the matrix that was previously passed to the 
     * {@link #setup(FloatMatrixDeviceCSR)} method.
     * 
     * @param x The solution
     * @param b The right hand side of the equation
     */
    void solve(Pointer x, Pointer b);
}