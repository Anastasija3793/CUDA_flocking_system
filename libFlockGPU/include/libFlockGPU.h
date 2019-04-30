#ifndef LIBFLOCKGPU_H
#define LIBFLOCKGPU_H

/// @file libFlockGPU.h
/// @brief Wrap-up library of parallelized program for creating fireflies flocking system
/// @author Anastasija Belaka
/// @version N/A
/// @date 30/04/2019 Updated to NCCA Coding standard
/// Revision History : https://github.com/Anastasija3793/CUDA_flocking_system
/// Initial Version 12/02/2019

class FlockGPU;
class libFlockGPU
{
public:
    libFlockGPU(int _numBoids);

    void separate();
    void align();
    void cohesion();

    void flock();

    void update();
    void dumpGeo(int _frameNumber);

    FlockGPU * m_flock;
};

#endif //LIBFLOCKGPU
