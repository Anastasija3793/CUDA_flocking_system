#ifndef BOIDGPUKERNELS_CUH
#define BOIDGPUKERNELS_CUH

#include <iostream>
#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <cutil_math.h>


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

#define NUM_BOIDS 100

//__global__ void updateKernel(float * _posX, float * _posY, float * _posZ, const float * _velX, const float * _velY, const float * _velZ)
//{
//    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

//    if(idx<NUM_BOIDS)
//    {
//        _posX[idx]+=_velX[idx];
//        _posY[idx]+=_velY[idx];
//        _posZ[idx]+=_velZ[idx];
//    }
//}


__global__ void updateKernel(float * _posX, float * _posY, float * _posZ, const float * _velX, const float * _velY, const float * _velZ)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<NUM_BOIDS)
    {
        _posX[idx]+=_velX[idx];
        _posY[idx]+=_velY[idx];
        _posZ[idx]+=_velZ[idx];
    }
}




//__device__ float lengthKernel(float3 _vec)
//{
//    float length = sqrtf((_vec.x*_vec.x)+(_vec.y*_vec.y)+(_vec.z*_vec.z));
//    return length;
//}

////change to __device__ later
//__global__ void steerKernel(float3 * _pos, float3 * _vel, float3 * _target, float3 * _steer)
//{
//    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
//    float3 desired[NUM_BOIDS];
//    float steerLength[NUM_BOIDS];

//    float max_speed = 1.0;
//    float max_force = 0.03;

//    desired[idx] = _target[idx] - _pos[idx];
//    //float normalize(desired[idx]);
//    desired[idx] = desired[idx] * max_speed;

//    _steer[idx] = desired[idx] - _vel[idx];
//    //m_steer.length()
//    steerLength[idx] = lengthKernel(_steer[idx]);

//    // Limit to maximum steering force (limit by max_force)
//    if(idx<NUM_BOIDS)
//    {
//        if(steerLength[idx] > max_force)
//        {
//            _steer[idx] = (_steer[idx]/steerLength[idx])*max_force;
//        }
//    }
//}



#endif // BOIDGPUKERNELS_CUH
