#include "Boid.h"
#include "Flock.h"
#include <ngl/Random.h>
#include <ngl/Transformation.h>
#include <ngl/ShaderLib.h>
#include <ngl/VAOPrimitives.h>
#include <math.h>

//#include <iostream>
#define PI = 3.14159


Boid::Boid(Flock *_flock)
{
    m_pos = ngl::Vec3((float(rand())/RAND_MAX), (float(rand())/RAND_MAX), (float(rand())/RAND_MAX));
    m_vel=ngl::Vec3((float(rand())/RAND_MAX), (float(rand())/RAND_MAX), (float(rand())/RAND_MAX));

    m_acc = ngl::Vec3(0.0f,0.0f,0.0f);
    max_speed = 1; //3
    max_force = 0.03; //0.05

    m_flock=_flock;
}

void Boid::applyForce(ngl::Vec3 _force)
{
    m_acc+=_force;
}

//void Boid::flock()
//{
//    //3 rules
//    //ngl::Vec3 sep = ngl::Vec3(0.0f,0.0f,0.0f);
//    //ngl::Vec3 ali = ngl::Vec3(0.0f,0.0f,0.0f);
//    //ngl::Vec3 coh = ngl::Vec3(0.0f,0.0f,0.0f);
//    //ngl::Vec3 sep = separate();
//    //ngl::Vec3 ali = align();
//    //ngl::Vec3 coh = cohesion();

//    //separate(sep);
//    //align(ali);
//    //cohesion(coh);

//    //sep*=1.5;
//    //ali*=0.02;
//    //coh*=1.0;

//    //applyForce(sep);
////    applyForce(ali);
//    //applyForce(coh);
//}


void Boid::seek(ngl::Vec3& _target)
{
    //m_target = _target;
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
    //_target = m_steer;
    //return m_steer;
}

void Boid::separate(ngl::Vec3& _sepVec)
{
    //m_steer = ngl::Vec3(0.0f,0.0f,0.0f);
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
        //m_steer/=count;
    }
    if(_sepVec.length()>0)
    {
        _sepVec.normalize();
        _sepVec*=max_speed;
        _sepVec-=m_vel;
        //limit by max_force
        if(_sepVec.length() > max_force)
        {
            _sepVec = (_sepVec/_sepVec.length())*max_force;
        }
    }
    //return m_steer;
}

void Boid::align(ngl::Vec3& _aliVec)
{
    //ngl::Vec3 sum = ngl::Vec3(0.0f,0.0f,0.0f);
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
    //return m_steer;
}

void Boid::cohesion(ngl::Vec3& _cohVec)
{
    //ngl::Vec3 sum = ngl::Vec3(0.0f,0.0f,0.0f);
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

void Boid::update()
{
    //flock();
    m_vel+=m_acc;

    // limit velocity by max_speed
    if(m_vel.length() > max_speed)
    {
        m_vel = (m_vel/m_vel.length())*max_speed;
    }
    m_pos+=m_vel;
    m_acc*=max_speed;
}
