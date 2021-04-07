#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif

//inline
//cudaError_t CUDA_CALL(cudaError_t result)
//{
//	if (result != cudaSuccess)
//		std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
//	return result;
//}