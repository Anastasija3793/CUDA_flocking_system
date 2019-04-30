#include "Boid.h"
#include "Flock.h"
#include <ngl/Random.h>
#include <ngl/Transformation.h>
#include <ngl/ShaderLib.h>
#include <ngl/VAOPrimitives.h>
#include <math.h>

#define PI = 3.14159

Boid::Boid(Flock *_flock)
{
    m_pos = ngl::Vec3((float(rand())/RAND_MAX), (float(rand())/RAND_MAX), (float(rand())/RAND_MAX));
    m_vel = ngl::Vec3((float(rand())/RAND_MAX), (float(rand())/RAND_MAX), (float(rand())/RAND_MAX));

    m_flock=_flock;
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief another constructor for testing
//----------------------------------------------------------------------------------------------------------------------
Boid::Boid(ngl::Vec3 _pos, ngl::Vec3 _vel)
{
    m_pos = _pos;
    m_vel = _vel;
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief another constructor for testing
//----------------------------------------------------------------------------------------------------------------------
Boid::Boid(ngl::Vec3 _pos, ngl::Vec3 _vel, Flock *_flock)
{
    m_pos = _pos;
    m_vel = _vel;

    m_flock=_flock;
}
//----------------------------------------------------------------------------------------------------------------------
void Boid::applyForce(ngl::Vec3 _force)
{
    m_acc+=_force;
}
//----------------------------------------------------------------------------------------------------------------------
void Boid::seek(ngl::Vec3& _target)
{
    m_target = _target;
    m_desired = _target - m_pos;
    // Scale to maximum speed
    m_desired.normalize();
    m_desired*=max_speed;

    _target = m_desired - m_vel;
    // Limit to maximum steering force (limit by max_force)
    if(_target.length() > max_force)
    {
        _target = (_target/_target.length())*max_force;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void Boid::separate(ngl::Vec3& _sepVec)
{
    int count = 0;
    // For every boid check if too close
    for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
    {
        float d = (m_pos - m_flock->m_boids[i].m_pos).length();
        if((d>0) && (d<m_sepRad))
        {
            ngl::Vec3 diff = m_pos - m_flock->m_boids[i].m_pos;
            diff.normalize();
            diff/=d;
            _sepVec+=diff;
            count++;
        }
    }
    // Average
    if(count>0)
    {
        _sepVec/=(float(count));
    }
    if(_sepVec.length()>0)
    {
        _sepVec.normalize();
        _sepVec*=max_speed;
        _sepVec-=m_vel;
        // Limit to maximum force
        if(_sepVec.length() > max_force)
        {
            _sepVec = (_sepVec/_sepVec.length())*max_force;
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
void Boid::align(ngl::Vec3& _aliVec)
{
    int count = 0;
    for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
    {
        float d = (m_pos - m_flock->m_boids[i].m_pos).length();
        if((d>0) && (d<m_neighbourDist))
        {
            _aliVec+=m_flock->m_boids[i].m_vel;
            count++;
        }
    }
    if(count>0)
    {
        _aliVec/=(float(count));

        _aliVec.normalize();
        _aliVec*=max_speed;
        _aliVec = _aliVec - m_vel;

    }
}
//----------------------------------------------------------------------------------------------------------------------
void Boid::cohesion(ngl::Vec3& _cohVec)
{
    int count = 0;
    for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
    {
        float d = (m_pos - m_flock->m_boids[i].m_pos).length();
        if((d>0) && (d<m_neighbourDist))
        {
            _cohVec+=m_flock->m_boids[i].m_pos;
            count++;
        }
    }
    if(count>0)
    {
        _cohVec/=count;

    }
    seek(_cohVec);
}
//----------------------------------------------------------------------------------------------------------------------
void Boid::update()
{
    m_vel+=m_acc;

    // Limit velocity to maximum speed
    if(m_vel.length() > max_speed)
    {
        m_vel = (m_vel/m_vel.length())*max_speed;
    }
    m_pos+=m_vel;
    m_acc*=max_speed;
}
//----------------------------------------------------------------------------------------------------------------------
