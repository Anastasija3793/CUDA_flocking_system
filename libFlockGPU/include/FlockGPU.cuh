#ifndef FLOCKGPU_CUH
#define FLOCKGPU_CUH

/// @file FlockGPU.cuh
/// @brief Library of parallelized program for creating a flock with its attributes and sets of flocking behaviour rules
/// @author Anastasija Belaka
/// @version N/A
/// @date 30/04/2019 Updated to NCCA Coding standard
/// Revision History : https://github.com/Anastasija3793/CUDA_flocking_system
/// Initial Version 12/02/2019

#include "iostream"
#include <random>
#include <algorithm>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

#include <stdlib.h>


class FlockGPU
{
public:
    FlockGPU(int _numBoids);
    ~FlockGPU();

    void separate();
    void align();
    void cohesion();

    void flock();

    void update();
    int randFloats(float *&devData, const size_t n);
    void dumpGeo(const uint _frameNumber);

//private:
    int m_numBoids;


    // stores boids position
    thrust::device_vector<float3> m_dPos;
    thrust::device_vector<float> m_dPosX;
    thrust::device_vector<float> m_dPosY;
    thrust::device_vector<float> m_dPosZ;
    float3 * m_dPosPtr;

    std::vector<float3> m_pos;

    // stores boids velocity
    thrust::device_vector<float3> m_dVel;
    thrust::device_vector<float> m_dVelX;
    thrust::device_vector<float> m_dVelY;
    thrust::device_vector<float> m_dVelZ;
    float3 * m_dVelPtr;

    thrust::device_vector<float3> m_dSep;
    thrust::device_vector<float> m_dSepX;
    thrust::device_vector<float> m_dSepY;
    thrust::device_vector<float> m_dSepZ;
    float3 * m_dSepPtr;

    thrust::device_vector<float3> m_dCoh;
    thrust::device_vector<float> m_dCohX;
    thrust::device_vector<float> m_dCohY;
    thrust::device_vector<float> m_dCohZ;
    float3 * m_dCohPtr;

    thrust::device_vector<float3> m_dAli;
    thrust::device_vector<float> m_dAliX;
    thrust::device_vector<float> m_dAliY;
    thrust::device_vector<float> m_dAliZ;
    float3 * m_dAliPtr;

    thrust::device_vector<float3> m_dAcc;
    thrust::device_vector<float> m_dAccX;
    thrust::device_vector<float> m_dAccY;
    thrust::device_vector<float> m_dAccZ;
    float3 * m_dAccPtr;
};

#endif // FLOCKGPU_CUH
