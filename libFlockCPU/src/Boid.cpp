#include "Boid.h"
#include "Flock.h"
#include <ngl/Random.h>
#include <ngl/Transformation.h>
#include <ngl/ShaderLib.h>
#include <ngl/VAOPrimitives.h>
#include <math.h>

//#include <iostream>
#define PI = 3.14159


Boid::Boid(/*ngl::Vec3 _pos, */Flock *_flock)
{
//    ngl::Random *rand=ngl::Random::instance();
//    ngl::Vec3 r = rand->getRandomVec3();

    m_pos = ngl::Vec3((float(rand())/RAND_MAX), (float(rand())/RAND_MAX), (float(rand())/RAND_MAX));
    //m_vel = r;
    m_vel=ngl::Vec3((float(rand())/RAND_MAX), (float(rand())/RAND_MAX), (float(rand())/RAND_MAX));

    m_acc = ngl::Vec3(0.0f,0.0f,0.0f);
    max_speed = 1; //3
    max_force = 0.03; //0.05

    m_flock=_flock;
    //collision radius for bbox
    //m_radius = 10;//3.0
    //hit value for bbox collision
    //m_hit=false;
}

//void Boid::updateRotation()
//{
//    //ngl::Vec3 facing = {0,0,1};
//    //rotation 0 when facing in z axis
//    ngl::Vec3 facing = ngl::Vec3(0.0f,0.0f,1.0f);
//    //ngl::Random *rand=ngl::Random::instance();

//    // if moving -> update
//    if(m_vel.operator != (ngl::Vec3(0.0f,0.0f,0.0f)))
//    //if(m_vel.operator !=({0,0,0}))
//    {
//        float mag1 = facing.length();
//        float mag2 = m_vel.length();

//        //angle between our axis (z) and boid vel vector
//        float steer = std::acos(facing.dot(m_vel)/(mag1*mag2));
//        //radians -> degrees
//        steer = steer*(180/M_PI);

//        //if rotation past 180 degrees must take away from 360, then update boid rotation
//        if(m_vel[0]>0)
//        {
//         m_rotation[1] = steer;
//        }
//        else
//        {
//         m_rotation[1]= 360-steer;
//        }
//    }
//}

void Boid::applyForce(ngl::Vec3 _force)
{
    m_acc+=_force;
}

//void Boid::bbox()
//{
//    ngl::Vec3 desiredVel;

//        if(m_pos.m_x >= 2 && m_vel.m_x >0)
//        {
//            desiredVel = ngl::Vec3(-m_vel.m_x,m_vel.m_y,m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox x bounds\n";
//        }
//        else if(m_pos.m_x <= -2 && m_vel.m_x <0)
//        {
//            desiredVel = ngl::Vec3(-m_vel.m_x,m_vel.m_y,m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox -x bounds\n";
//        }
//        else if(m_pos.m_y >= 2 && m_vel.m_y >0)
//        {
//            desiredVel = ngl::Vec3(m_vel.m_x,-m_vel.m_y,m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox y bounds\n";
//        }
//        else if(m_pos.m_y <= -2 && m_vel.m_y <0)
//        {
//            desiredVel = ngl::Vec3(m_vel.m_x,-m_vel.m_y,m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox -y bounds\n";
//        }
//        else if(m_pos.m_z >= 2 && m_vel.m_z >0)
//        {
//            desiredVel = ngl::Vec3(m_vel.m_x,m_vel.m_y,-m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox z bounds\n";
//        }
//        else if(m_pos.m_z <= -2 && m_vel.m_z <0)
//        {
//            desiredVel = ngl::Vec3(m_vel.m_x,m_vel.m_y,-m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox -z bounds\n";
//        }
//}

void Boid::flock()
{
    //3 rules
    ngl::Vec3 sep = separate();
    ngl::Vec3 ali = align();
    ngl::Vec3 coh = cohesion();

    sep*=1.5;
    ali*=0.02;
    coh*=1.0;

    applyForce(sep);
//    applyForce(ali);
    applyForce(coh);
}


ngl::Vec3 Boid::seek(ngl::Vec3 _target)
{
    m_target = _target;
    m_desired = m_target - m_pos;
    // Scale to maximum speed
    m_desired.normalize();
    m_desired*=max_speed;

    m_steer = m_desired - m_vel;
    // Limit to maximum steering force (limit by max_force)
    if(m_steer.length() > max_force)
    {
        m_steer = (m_steer/m_steer.length())*max_force;
    }
    return m_steer;
}

ngl::Vec3 Boid::separate()
{
    m_steer = ngl::Vec3(0.0f,0.0f,0.0f);
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
            m_steer+=diff;
            count++;
        }
    }
    // Average
    if(count>0)
    {
        m_steer/=(float(count));
        //m_steer/=count;
    }
    if(m_steer.length()>0)
    {
        m_steer.normalize();
        m_steer*=max_speed;
        m_steer-=m_vel;
        //limit by max_force
        if(m_steer.length() > max_force)
        {
            m_steer = (m_steer/m_steer.length())*max_force;
        }
    }
    return m_steer;
}

ngl::Vec3 Boid::align()
{
    ngl::Vec3 sum = ngl::Vec3(0.0f,0.0f,0.0f);
    int count = 0;
    for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
    {
        float d = (m_pos - m_flock->m_boids[i].m_pos).length();
        if((d>0) && (d<m_neighbourDist))
        {
            sum+=m_flock->m_boids[i].m_vel;
            count++;
        }
    }
    if(count>0)
    {
        sum/=(float(count));

        sum.normalize();
        sum*=max_speed;
        m_steer = sum - m_vel;

    }
    return m_steer;
//    else
//    {
//        return ngl::Vec3(0.0f,0.0f,0.0f);
//    }
}

ngl::Vec3 Boid::cohesion()
{
    ngl::Vec3 sum = ngl::Vec3(0.0f,0.0f,0.0f);
    int count = 0;
    for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
    {
        float d = (m_pos - m_flock->m_boids[i].m_pos).length();
        if((d>0) && (d<m_neighbourDist))
        {
            sum+=m_flock->m_boids[i].m_pos;
            count++;
        }
    }
    if(count>0)
    {
        sum/=count;

    }
    return seek(sum);
//    else
//    {
//        return ngl::Vec3(0.0f,0.0f,0.0f);
//    }
}


//void Boid::loadMatricesToShader(ngl::Transformation &_tx, const ngl::Mat4 &_globalMat, const ngl::Mat4 &_view, const ngl::Mat4 &_project) const
//{
//    ngl::ShaderLib *shader=ngl::ShaderLib::instance();

//    ngl::Mat4 MV;
//    ngl::Mat4 MVP;
//    ngl::Mat3 normalMatrix;
//    MV=_view  *_globalMat* _tx.getMatrix();

//    MVP=_project*MV;
//    normalMatrix=MV;
//    normalMatrix.inverse().transpose();
//    shader->setUniform("MVP",MVP);
//    shader->setUniform("normalMatrix",normalMatrix);
//}

//void Boid::draw(const std::string &_shaderName, const ngl::Mat4 &_globalMat, const ngl::Mat4 &_view, const ngl::Mat4 &_project) const
//{
//    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
//    shader->use(_shaderName);
//    // grab an instance of the primitives for drawing
//    ngl::VAOPrimitives *prim=ngl::VAOPrimitives::instance();
//    ngl::Transformation t;

//    t.setPosition(m_pos);
//    t.setRotation(m_rotation);
//    loadMatricesToShader(t,_globalMat,_view,_project);
//    //prim->draw("cone");
//    prim->draw("sphere");
//}

void Boid::update()
{
    //m_vel+=m_acc;
    //m_pos+=m_vel;
    //BBoxCollision();
//    bbox();
    flock();
    m_vel+=m_acc;


    // if statement for limiting velocity by max speed
//       auto NormV = m_vel;
//       auto speed = m_vel.length();
//       if (speed > max_speed)
//       {
//           NormV.normalize();
//           m_vel = NormV * max_speed;
//       }


    // limit velocity by max_speed
    if(m_vel.length() > max_speed)
    {
        m_vel = (m_vel/m_vel.length())*max_speed;
    }
    m_pos+=m_vel;
    m_acc*=max_speed;

//    updateRotation();
    //m_hit=false;
}
