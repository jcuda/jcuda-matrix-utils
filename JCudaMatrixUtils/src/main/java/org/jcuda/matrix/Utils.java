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

package org.jcuda.matrix;

import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

/**
 * Utility methods for memory handling
 */
public class Utils
{
    /**
     * Create an array from the given device data with the given size
     *
     * @param deviceData The device data
     * @param size The size (in number of elements)
     * @return The new array
     */
    public static float[] createFloatArray(Pointer deviceData, int size)
    {
        float hostData[] = new float[size];
        cudaMemcpy(Pointer.to(hostData), deviceData,
            size * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        return hostData;
    }

    /**
     * Create an array from the given device data with the given size
     *
     * @param deviceData The device data
     * @param size The size (in number of elements)
     * @return The new array
     */
    public static double[] createDoubleArray(Pointer deviceData, int size)
    {
        double hostData[] = new double[size];
        cudaMemcpy(Pointer.to(hostData), deviceData,
            size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        return hostData;
    }
    
    /**
     * Create an array from the given device data with the given size
     *
     * @param deviceData The device data
     * @param size The size (in number of elements)
     * @return The new array
     */
    public static int[] createIntArray(Pointer deviceData, int size)
    {
        int hostData[] = new int[size];
        cudaMemcpy(Pointer.to(hostData), deviceData,
            size * Sizeof.INT, cudaMemcpyDeviceToHost);
        return hostData;
    }

    /**
     * Create a pointer to device memory that has the same contents
     * as the given array.
     *
     * @param hostData The host data
     * @return The new pointer
     */
    public static CUdeviceptr createPointer(float hostData[])
    {
        CUdeviceptr deviceData = new CUdeviceptr();
        cudaMalloc(deviceData, hostData.length * Sizeof.FLOAT);
        cudaMemcpy(deviceData, Pointer.to(hostData),
            hostData.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        return deviceData;
    }

    /**
     * Create a pointer to device memory that has the same contents
     * as the given array.
     *
     * @param hostData The host data
     * @return The new pointer
     */
    public static CUdeviceptr createPointer(double hostData[])
    {
        CUdeviceptr deviceData = new CUdeviceptr();
        cudaMalloc(deviceData, hostData.length * Sizeof.DOUBLE);
        cudaMemcpy(deviceData, Pointer.to(hostData),
            hostData.length * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
        return deviceData;
    }

    /**
     * Create a pointer to device memory that has the same contents
     * as the given array.
     *
     * @param hostData The host data
     * @return The new pointer
     */
    public static CUdeviceptr createPointer(int hostData[])
    {
        CUdeviceptr deviceData = new CUdeviceptr();
        cudaMalloc(deviceData, hostData.length * Sizeof.INT);
        cudaMemcpy(deviceData, Pointer.to(hostData),
            hostData.length * Sizeof.INT, cudaMemcpyHostToDevice);
        return deviceData;
    }

    /**
     * Copy the contents of the given array to the given device pointer
     *
     * @param hostData The host data
     * @param deviceData The device data
     */
    public static void copyToDevice(float hostData[], Pointer deviceData)
    {
        cudaMemcpy(deviceData, Pointer.to(hostData),
            hostData.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);
    }

    /**
     * Copy the contents of the given array to the given device pointer
     *
     * @param hostData The host data
     * @param deviceData The device data
     */
    public static void copyToDevice(double hostData[], Pointer deviceData)
    {
        cudaMemcpy(deviceData, Pointer.to(hostData),
            hostData.length * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
    }

    /**
     * Copy the contents of the given device pointer to the given array
     *
     * @param deviceData The device data
     * @param hostData The host data
     */
    public static void copyToHost(Pointer deviceData, float hostData[])
    {
        cudaMemcpy(Pointer.to(hostData), deviceData,
            hostData.length * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
    }
    
    /**
     * Copy the contents of the given device pointer to the given array
     *
     * @param deviceData The device data
     * @param hostData The host data
     */
    public static void copyToHost(Pointer deviceData, double hostData[])
    {
        cudaMemcpy(Pointer.to(hostData), deviceData,
            hostData.length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
    }



    /**
     * Private constructor to prevent instantiation
     */
    private Utils()
    {
        // Private constructor to prevent instantiation
    }
}
