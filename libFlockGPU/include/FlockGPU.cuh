#ifndef FLOCKGPU_CUH
#define FLOCKGPU_CUH

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

private:
    int m_numBoids;

    // stores boids x,y,z, position //float3?
    thrust::device_vector<float3> m_dPos;
    thrust::device_vector<float> m_dPosX;
    thrust::device_vector<float> m_dPosY;
    thrust::device_vector<float> m_dPosZ;

    float3 * m_dPosPtr;
//    float * m_dPosXPtr;

//    thrust::device_vector<float> getposX;
//    thrust::device_vector<float> getposY;
//    thrust::device_vector<float> getposZ;
//    float * m_dPosYPtr;
//    float * m_dPosZPtr;
    std::vector<float3> m_pos;
    std::vector<float> xTest;
    std::vector<float> yTest;
    std::vector<float> zTest;
    //float m_dPos;
    std::vector<float3> m_test;
    std::vector<float3> m_testVel;

    // stores boids velocity
    thrust::device_vector<float3> m_dVel;
    thrust::device_vector<float> m_dVelX;
    thrust::device_vector<float> m_dVelY;
    thrust::device_vector<float> m_dVelZ;

    float3 * m_dVelPtr;
//    float * m_dVelXPtr;
//    float * m_dVelYPtr;
//    float * m_dVelZPtr;

    thrust::device_vector<float3> m_dSep;
    //test
    thrust::device_vector<float> m_dSepX;
    thrust::device_vector<float> m_dSepY;
    thrust::device_vector<float> m_dSepZ;

    float3 * m_dSepPtr;
//    std::vector<float3> m_sep;

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
