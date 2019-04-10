#include "BoidGPU.cuh"
#include <iostream>

//#define PI = 3.14159

BoidGPU::BoidGPU(thrust::device_vector<float> _pos, FlockGPU *_flock)
{
    m_pos.resize(3,0);
    m_vel.resize(3,0);

    m_flock = _flock;
}

//BoidGPU::~BoidGPU()
//{
    
//}
