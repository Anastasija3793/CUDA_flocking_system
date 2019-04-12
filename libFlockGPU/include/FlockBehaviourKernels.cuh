#ifndef FLOCKBEHAVIOURKERNELS_CUH
#define FLOCKBEHAVIOURKERNELS_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>



//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
//#endif

#define NUM_BOIDS 100


__device__ void steerKernel(float3 * _pos, float3 * _target, float3 * _steer)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    float3 desired[NUM_BOIDS];

    desired[idx] = _target[idx] - _pos[idx];
    //desired.normalize
    //desired*=max_speed

//    m_steer = m_desired - m_vel;
//        // Limit to maximum steering force (limit by max_force)
//    if(m_steer.length() > max_force)
//    {
//        m_steer = (m_steer/m_steer.length())*max_force;
//    }
    //_steer[idx] = steer/
}



#endif // FLOCKBEHAVIOURKERNELS_CUH
