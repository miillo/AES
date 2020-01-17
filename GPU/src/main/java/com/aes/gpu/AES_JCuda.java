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


public class AES_JCuda {

    public void executeGPUTestsTest() {
        printGPUInfo();
    }


    public static byte galoisMulGPU(byte a, byte b) {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file todo ptx path to params
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "D:\\Coding\\Java\\AES\\GPU\\src\\main\\resources\\galoisMul.ptx");

        // Obtain a function pointer to the "add" function
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "galoisMul");


        char ch1 = (char)a;
        char ch2 = (char)b;
        char[] hostInputA = {ch1};
        char[] hostInputB = {ch2};

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, 2 * Sizeof.CHAR);
        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA), 2 * Sizeof.CHAR);
        CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, 2 * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB), 2 * Sizeof.FLOAT);

        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, 2 * Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{2}),
                Pointer.to(deviceInputA),
                Pointer.to(deviceInputB),
                Pointer.to(deviceOutput)
        );

        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)2 / blockSizeX);
        cuLaunchKernel(function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        char[] hostOutput = new char[2];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
                2 * Sizeof.CHAR);

        System.out.println(Arrays.toString(hostOutput));
        System.out.println((byte)hostOutput[0] & 0xff);

        // Clean up.
        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);

        return (byte)1;
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
