#include "FlockGPU.cuh"
#include "libFlockGPU.h"

libFlockGPU::libFlockGPU(int _numBoids)
{
    m_flock = new FlockGPU(_numBoids);
}

void libFlockGPU::update()
{
    m_flock->update();
}

void libFlockGPU::dumpGeo(int _frameNumber)
{
    m_flock->dumpGeo(_frameNumber);
}
