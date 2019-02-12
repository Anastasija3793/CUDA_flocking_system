#include "Boid.h"
#include <ngl/Random.h>
#include <ngl/Transformation.h>
#include <ngl/ShaderLib.h>
#include <ngl/VAOPrimitives.h>
#include <math.h>

#define PI = 3.14159


Boid::Boid(ngl::Vec3 _pos, ngl::Vec3 _vel, Flock *_flock)
{
    m_pos=_pos;
    m_vel=_vel;
    //m_target=_target;
    m_flock=_flock;
}

void Boid::updateRotation()
{
    //ngl::Vec3 facing = {0,0,1};
    //rotation 0 when facing in z axis
    ngl::Vec3 facing = ngl::Vec3(0.0f,0.0f,1.0f);
    ngl::Random *rand=ngl::Random::instance();


//    if(m_vel.m_x > m_vel.m_y)
//    {
//        facing = rand->getRandomVec3();
//    }
//    if(m_vel.m_x >= m_vel.m_y)
//    {
//        facing.set(0.0f,1.0f,0.0f);
//    }

    // if moving -> update
    if(m_vel.operator != (ngl::Vec3(0.0f,0.0f,0.0f)))
    //if(m_vel.operator !=({0,0,0}))
    {
        float mag1 = facing.length();
        float mag2 = m_vel.length();

        //angle between our axis (z) and boid vel vector
        float steer = std::acos(facing.dot(m_vel)/(mag1*mag2));
        //radians -> degrees
        steer = steer*(180/M_PI);

        //if rotation past 180 degrees must take away from 360, then update boid rotation
        if(m_vel[0]>0)
        {
         m_rotation[1] = steer;
        }
        else
        {
         m_rotation[1]= 360-steer;
        }
    }
}

void Boid::loadMatricesToShader(ngl::Transformation &_tx, const ngl::Mat4 &_globalMat, const ngl::Mat4 &_view, const ngl::Mat4 &_project) const
{
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();

    ngl::Mat4 MV;
    ngl::Mat4 MVP;
    ngl::Mat3 normalMatrix;
    MV=_view  *_globalMat* _tx.getMatrix();

//    MV.rotateX(360);
//    MV.rotateY(360);
//    MV.rotateZ(360);

    MVP=_project*MV;
    normalMatrix=MV;
    normalMatrix.inverse().transpose();
    shader->setUniform("MVP",MVP);
    shader->setUniform("normalMatrix",normalMatrix);
}

void Boid::draw(const std::string &_shaderName, const ngl::Mat4 &_globalMat, const ngl::Mat4 &_view, const ngl::Mat4 &_project) const
{
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    shader->use(_shaderName);
    // grab an instance of the primitives for drawing
    ngl::VAOPrimitives *prim=ngl::VAOPrimitives::instance();
    ngl::Transformation t;

    t.setPosition(m_pos);
    t.setRotation(m_rotation);
    loadMatricesToShader(t,_globalMat,_view,_project);
    prim->draw("cone");
}

void Boid::move()
{
    ngl::Random *rand=ngl::Random::instance();
    ngl::Vec3 randVel = rand->getRandomVec3();
//    if(m_pos.m_x >= 10)
//    {
//        for(int i=0; i<10; i++)
//        {
//            //m_vel = m_vel*(-1);
//            m_vel.m_x -= 0.02;
//        }

//    }
    m_pos+=m_vel;
    m_vel+=randVel;
    updateRotation();
}
