#ifndef LIBFLOCKGPU_H
#define LIBFLOCKGPU_H

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
