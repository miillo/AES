package com.aes.gpu;

import com.aes.AES_GPU;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;

import java.util.Arrays;


public class JCudaTest {

    /**
     * Prints GPU info
     */
    public static void printGPUInfo() {
        JCuda.setExceptionsEnabled(true);
        int[] deviceCount = {0};
        cudaGetDeviceCount(deviceCount);
        System.out.println("Found " + deviceCount[0] + " devices");
        for (int device = 0; device < deviceCount[0]; device++) {
            System.out.println("Properties of device " + device + ":");
            cudaDeviceProp deviceProperties = new cudaDeviceProp();
            cudaGetDeviceProperties(deviceProperties, device);
            System.out.println(deviceProperties.toFormattedString());
        }
    }
}
