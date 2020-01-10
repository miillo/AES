package com.aes.gpu;

import com.aes.AES_GPU;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;


public class AES_JCuda {

    public void executeGPUTestsTest() {
        printGPUInfo();
    }

    /**
     * Prints GPU info
     */
    private void printGPUInfo() {
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
