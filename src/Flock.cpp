#include "Flock.h"
#include <ngl/Random.h>

Flock::Flock(ngl::Vec3 _pos, int _numBoids)
{
    ngl::Random *rand=ngl::Random::instance();
    auto randomVelocity = rand->getRandomVec3();
    for (int i=0; i< _numBoids; ++i)
    {
        m_boids.push_back(Boid(_pos,randomVelocity,this));
        randomVelocity = rand->getRandomVec3();
    }
    m_numBoids=_numBoids;
}

Flock::~Flock()
{
    //dctor
}

void Flock::move()
{
    for(int i=0; i<m_numBoids; ++i)
    {
        m_boids[i].move();
    }
}

void Flock::draw(const std::string &_shaderName, const ngl::Mat4 &_globalMat, const ngl::Mat4 &_view, const ngl::Mat4 &_project) const
{
    for(int i=0; i<m_numBoids; ++i)
    {
        m_boids[i].draw(_shaderName, _globalMat, _view, _project);
    }
}
