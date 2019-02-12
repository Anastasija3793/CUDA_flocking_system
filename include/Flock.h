#ifndef FLOCK_H
#define FLOCK_H

#include <vector>
#include <ngl/Vec3.h>
#include "Boid.h"

class Flock
{
public:
    Flock(ngl::Vec3 _pos, int _numBoids);
    ~Flock();
    void draw(const std::string &_shaderName,const ngl::Mat4 &_globalMat, const  ngl::Mat4 &_view, const ngl::Mat4 &_project)const ;
    void move();

    int m_numBoids;

private:
    ngl::Vec3 m_pos;
    std::vector<Boid>m_boids;
    //ngl::Vec3 m_vel;
    //ngl::Vec3 m_target;
};

#endif // FLOCK_H
