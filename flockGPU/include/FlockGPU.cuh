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
    //void draw();
    void dumpGeo(const uint _frameNumber);

private:
    int m_numBoids;

    // stores boids x,y,z, position
    thrust::device_vector<float> m_dPosX;
    thrust::device_vector<float> m_dPosY;
    thrust::device_vector<float> m_dPosZ;

    //float * m_dPosX_ptr;
    //float * m_dPosY_ptr;
    //float * m_dPosZ_ptr;

    // stores boids velocity
    thrust::device_vector<float> m_dVelX;
    thrust::device_vector<float> m_dVelY;
    thrust::device_vector<float> m_dVelZ;

    //float * m_dVelX_ptr;
    //float * m_dVelY_ptr;
    //float * m_dVelZ_ptr;
};

#endif // FLOCKGPU_H
