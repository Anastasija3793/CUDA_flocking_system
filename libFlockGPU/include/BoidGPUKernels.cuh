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

__device__ float lengthKernel(float3 _vec)
{
    float length = sqrtf((_vec.x*_vec.x)+(_vec.y*_vec.y)+(_vec.z*_vec.z));
    return length;
}

__global__ void updateKernel(float3 * _pos, float3 * _acc, float3 * _vel)
{
    float velLength[NUM_BOIDS];
    float max_speed = 1.0; //1.0  //0.7

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<NUM_BOIDS)
    {
        _vel[idx]+=_acc[idx];
        velLength[idx] = lengthKernel(_vel[idx]);
        if(velLength[idx] > max_speed)
        {
            _vel[idx] = (_vel[idx]/velLength[idx])*max_speed;
        }
        _pos[idx]+=_vel[idx];
        _acc[idx]*=max_speed;
    }
}

__device__ void applyForceKernel(float3 * _force, float3 * _acc)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<NUM_BOIDS)
    {
        _acc[idx]+=_force[idx];
    }
}

//change to __device__ later
__device__ void seekKernel(const float3 * _pos, const float3 * _vel, const float3 * _target, float3 * _seek)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    float3 desired[NUM_BOIDS];
    float seekLength[NUM_BOIDS];
    // for desired.normalize()
    float desiredLength[NUM_BOIDS];

    float max_speed = 1.0; //1.0  //0.7 //10.0
    float max_force = 0.03; //0.03 //0.1 //1.0

    desired[idx] = _target[idx] - _pos[idx];

    //float normalize(desired[idx]);
    desiredLength[idx] = lengthKernel(desired[idx]);
    desired[idx] = (desired[idx]/desiredLength[idx]) * max_speed;

    _seek[idx] = desired[idx] - _vel[idx];
    //m_steer.length()
    seekLength[idx] = lengthKernel(_seek[idx]);

    // Limit to maximum steering force (limit by max_force)
    if(idx<NUM_BOIDS)
    {
        if(seekLength[idx] > max_force)
        {
            _seek[idx] = (_seek[idx]/seekLength[idx])*max_force;
        }
    }
}

__device__ void separateKernel(float3 * _sepVec, float3 * _pos, float3 * _vel)
{
    //m_steer = sepVec
    // for finding sepVec
    __shared__ float3 _diff[NUM_BOIDS];

    float diffLength[NUM_BOIDS];
    float sepVecLength[NUM_BOIDS];

    // for checking distance
    __shared__ float3 _dist[NUM_BOIDS];

    //count of neighbours
    __shared__ unsigned int count[NUM_BOIDS];
    float distLength[NUM_BOIDS];

    float max_speed = 1.0; //1.0
    float max_force = 0.03; //0.03

    // for current boid
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for current boid's neighbours
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    //for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
    if(idx < NUM_BOIDS && idy < NUM_BOIDS)
    {
        count[idx] = 0;
        _diff[idx].x = 0;
        _diff[idx].y = 0;
        _diff[idx].z = 0;
        __syncthreads();

        _dist[idx] = _pos[idx] - _pos[idy];

        distLength[idx] = lengthKernel(_dist[idx]);

        if((distLength[idx]>0)&&(distLength[idx]<15.0))
        {
            // instead of _diff[idx] = _pos[idx] - _pos[idy]; + _dist[idx]
            atomicAdd(&(_diff[idx].x), _dist[idx].x);
            atomicAdd(&(_diff[idx].y), _dist[idx].y);
            atomicAdd(&(_diff[idx].z), _dist[idx].z);
//            _diff[idx]+=_dist[idx];

            diffLength[idx] = lengthKernel(_diff[idx]);
            //(diff.normalize())/distLength;
            _diff[idx] = (_diff[idx]/diffLength[idx])/distLength[idx];

            //sepVec+=diff;
            atomicAdd(&(_sepVec[idx].x), _diff[idx].x);
            atomicAdd(&(_sepVec[idx].y), _diff[idx].y);
            atomicAdd(&(_sepVec[idx].z), _diff[idx].z);
//            _sepVec[idx]+=_diff[idx];

            //count++
            atomicAdd(&count[idx], 1);
//            count[idx]+=1;
        }
    }
    //need to sync
    __syncthreads();

    // Average
    //if(idy == 0 && idx< NUM_BOIDS)
    if(count[idx] > 0)
    {
        //m_steer/=(float(count));
        _sepVec[idx] = _sepVec[idx]/count[idx];
    }

    sepVecLength[idx] = lengthKernel(_sepVec[idx]);
    if(sepVecLength[idx] > 0)
    {
        //sepVec.normalize();
        _sepVec[idx] = (_sepVec[idx]/sepVecLength[idx])*max_speed;
        _sepVec[idx] = _sepVec[idx] - _vel[idx];

        //limit by max_force
        if(sepVecLength[idx] > max_force)
        {
            _sepVec[idx] = (_sepVec[idx]/sepVecLength[idx])*max_force;
        }
    }
}

__device__ void cohesionKernel(float3 * _cohVec, float3 * _pos, float3 * _vel)
{
    // for checking distance
    __shared__ float3 _dist[NUM_BOIDS];

    //count of neighbours
    __shared__ unsigned int count[NUM_BOIDS];
    float distLength[NUM_BOIDS];

    //float max_speed = 50.0; //1.0
    //float max_force = 5.0; //0.03

    // for current boid
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for current boid's neighbours
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    //for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
    if(idx < NUM_BOIDS && idy < NUM_BOIDS)
    {
        count[idx] = 0;
        __syncthreads();

        _dist[idx] = _pos[idx] - _pos[idy];
        distLength[idx] = lengthKernel(_dist[idx]);

        if((distLength[idx]>0)&&(distLength[idx]<30.0))
        {
            //cohVec += other pos;
            atomicAdd(&(_cohVec[idx].x), _pos[idy].x);
            atomicAdd(&(_cohVec[idx].y), _pos[idy].y);
            atomicAdd(&(_cohVec[idx].z), _pos[idy].z);
//            _cohVec[idx]+=_pos[idy];

            //count++
            atomicAdd(&count[idx], 1);
//            count[idx]+=1;
        }
    }
    //need to sync
    __syncthreads();

    // Average
    //if(idy == 0 && idx< NUM_BOIDS)
    if(count[idx] > 0)
    {
        //m_steer/=(float(count));
        _cohVec[idx] = _cohVec[idx]/count[idx];
    }
    seekKernel(_pos,_vel,_cohVec,_cohVec);
}

__device__ void alignmentKernel(float3 * _aliVec, float3 * _pos, float3 * _vel)
{
    // for checking distance
    __shared__ float3 _dist[NUM_BOIDS];

    //count of neighbours
    __shared__ unsigned int count[NUM_BOIDS];
    float distLength[NUM_BOIDS];
    float aliVecLength[NUM_BOIDS];

    float max_speed = 1.0; //1.0
//    float max_force = 2.0; //0.03

    // for current boid
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for current boid's neighbours
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    //for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
    if(idx < NUM_BOIDS && idy < NUM_BOIDS)
    {
        count[idx] = 0;
        __syncthreads();

        _dist[idx] = _pos[idx] - _pos[idy];
        distLength[idx] = lengthKernel(_dist[idx]);

        if((distLength[idx]>0)&&(distLength[idx]<30.0))
        {
            //cohVec += other pos;
            atomicAdd(&(_aliVec[idx].x), _vel[idy].x);
            atomicAdd(&(_aliVec[idx].y), _vel[idy].y);
            atomicAdd(&(_aliVec[idx].z), _vel[idy].z);
//            _aliVec[idx]+=_vel[idy];

            //count++
            atomicAdd(&count[idx], 1);
//            count[idx]+=1;
        }
    }
    //need to sync
    __syncthreads();

    aliVecLength[idx] = lengthKernel(_aliVec[idx]);
    // Average
    //if(idy == 0 && idx< NUM_BOIDS)
    if(count[idx] > 0)
    {
        //m_steer/=(float(count));
        _aliVec[idx] = _aliVec[idx]/count[idx];

        _aliVec[idx] = (_aliVec[idx]/aliVecLength[idx])*max_speed;
        _aliVec[idx] = _aliVec[idx] - _vel[idx];
    }
}

__global__ void flockKernel(float3 * _sepVec, float3 * _cohVec, float3 * _aliVec, float3 * _acc, float3 * _pos, float3 * _vel)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx<NUM_BOIDS)
    {
        separateKernel(_sepVec, _pos, _vel);
        cohesionKernel(_cohVec, _pos, _vel);
        alignmentKernel(_aliVec, _pos, _vel);

        __syncthreads();

        applyForceKernel(_sepVec,_acc);
        applyForceKernel(_cohVec,_acc);
//        applyForceKernel(_aliVec,_acc);

        if(idy == 0)
        {
            // sum 3 rules (later)
            //_vel[idx] = _vel[idx] + _sepVec[idx];
            _vel[idx] += (_sepVec[idx]*1.5) + (_cohVec[idx]*1) + (_aliVec[idx]*0.02);//1
        }
    }
}



#endif // BOIDGPUKERNELS_CUH
