#ifndef FLOCK_H
#define FLOCK_H

#include <vector>
#include <ngl/Vec3.h>
#include "Boid.h"

class Flock
{
public:
    Flock(int _numBoids);
    ~Flock();

    void separate();
    void align();
    void cohesion();

    void flock();
    void update();
    void dumpGeo(const uint _frameNumber);

    int m_numBoids;
    std::vector<Boid>m_boids;
};

#endif // FLOCK_H
