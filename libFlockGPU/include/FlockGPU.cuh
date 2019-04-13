#ifndef FLOCKGPU_CUH
#define FLOCKGPU_CUH

#include "iostream"
#include <random>
#include <algorithm>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>


class FlockGPU
{
public:
    FlockGPU(int _numBoids);
    ~FlockGPU();

    void update();
    int randFloats(float *&devData, const size_t n);
    //void draw();
    void dumpGeo(const uint _frameNumber);

private:
    int m_numBoids;

    // stores boids x,y,z, position //float3?
    thrust::device_vector<float3> m_dPos;
//    thrust::device_vector<float> m_dPosY;
//    thrust::device_vector<float> m_dPosZ;

    float3 * m_dPosPtr;
//    float * m_dPosYPtr;
//    float * m_dPosZPtr;

    // stores boids velocity
    thrust::device_vector<float3> m_dVel;
//    thrust::device_vector<float> m_dVelY;
//    thrust::device_vector<float> m_dVelZ;

    float3 * m_dVelPtr;
//    float * m_dVelYPtr;
//    float * m_dVelZPtr;

    thrust::device_vector<float3> m_dTarget;
    float3 * m_dTargetPtr;
    thrust::device_vector<float3> m_dSteer;
    float3 * m_dSteerPtr;
};

#endif // FLOCKGPU_H
