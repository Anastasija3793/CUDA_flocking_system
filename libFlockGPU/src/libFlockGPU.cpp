#include "FlockGPU.cuh"
#include "libFlockGPU.h"

libFlockGPU::libFlockGPU(int _numBoids)
{
    m_flock = new FlockGPU(_numBoids);
}

void libFlockGPU::separate()
{
    m_flock->separate();
}

void libFlockGPU::align()
{
    m_flock->align();
}

void libFlockGPU::cohesion()
{
    m_flock->cohesion();
}

void libFlockGPU::flock()
{
    m_flock->flock();
}

void libFlockGPU::update()
{
    m_flock->update();
}

void libFlockGPU::dumpGeo(int _frameNumber)
{
    m_flock->dumpGeo(_frameNumber);
}
