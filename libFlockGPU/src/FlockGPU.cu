#include "FlockGPU.cuh"
#include <cuda.h>

#include <iostream>

//cuda_call(x)...

FlockGPU::FlockGPU(int _numBoids)
{
    m_numBoids=_numBoids;

    m_dPosX.resize(m_numBoids);
    m_dPosY.resize(m_numBoids);
    m_dPosZ.resize(m_numBoids);
}

FlockGPU::~FlockGPU()
{

}

void FlockGPU::update()
{
    //update flock
}
