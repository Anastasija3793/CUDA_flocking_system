#include "Boid.h"
#include <ngl/Random.h>
#include <ngl/Transformation.h>
#include <ngl/ShaderLib.h>
#include <ngl/VAOPrimitives.h>
#include <math.h>

//#include <iostream>
#define PI = 3.14159


//Boid::Boid(ngl::Vec3 _pos, ngl::Vec3 _vel, Flock *_flock)
Boid::Boid(Flock *_flock)
{
//    ngl::Random *rand=ngl::Random::instance();
//    ngl::Vec3 randPos = rand->getRandomVec3();
//    ngl::Vec3 randVel = rand->getRandomVec3();
    //m_pos=_pos;
    //m_vel=_vel;
    //m_pos=ngl::Vec3(0.0f,0.0f,0.0f);
    //m_vel=ngl::Vec3(1.0f,0.0f,0.0f);
    m_pos=ngl::Vec3((float(rand())/RAND_MAX), (float(rand())/RAND_MAX), (float(rand())/RAND_MAX));
    m_vel=ngl::Vec3((float(rand())/RAND_MAX), (float(rand())/RAND_MAX), (float(rand())/RAND_MAX));
//    m_pos = randPos;
//    m_vel = randVel;
    //m_target=_target;
    m_flock=_flock;
    m_radius = 3.0f;

    m_acc = ngl::Vec3(0.0f,0.0f,0.0f);
    max_speed = 4; //100
    max_force = 0.03;

    m_hit=false;
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

void Boid::applyForce(ngl::Vec3 _force)
{
    m_acc+=_force;
}

void Boid::seek(ngl::Vec3 _target)
{
    m_target=_target;
    m_desired = m_target - m_pos;
    m_desired.normalize();

    m_desired*= max_speed;
    m_steer = m_desired - m_vel;
    m_steer.normalize();

    // if statement for limiting steering by max force
    auto NormS = m_steer;
    auto speed = m_steer.length();
    if (speed > max_force)
    {
        NormS.normalize();
        m_steer = NormS * max_force;
    }
    // applying the force to the steering
    applyForce(m_steer);//remove when using 3 rules
}

//void Boid::sepForce()
//{
//    for(int i = 0; i < m_sep.size(); ++i)
//    {
//        //applyForce(m_steer);
//    }
//}

/*void Boid::separate()
{
    for(int i = 0; i < m_sep.size(); ++i)
    {
        //m_steer = m_sep[i]->m_steer*200;
        m_vel.m_x = m_sep[i]->m_vel.m_x*=(-1);
        m_vel.m_y = m_sep[i]->m_vel.m_y*=(-1);
        m_vel.m_z = m_sep[i]->m_vel.m_z*=(-1);
        //m_vel = m_sep[i]->m_vel.operator *=(ngl::Vec3(0.0f,0.0f,0.0f));
        //m_pos = m_sep[i]->m_pos= ngl::Vec3(-20,0,0);
        //applyForce();
    }
}*/

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
    //ngl::Random *rand=ngl::Random::instance();
    //ngl::Vec3 randVel = rand->getRandomVec3();
//    if(m_pos.m_x >= 10)
//    {
//        for(int i=0; i<10; i++)
//        {
//            //m_vel = m_vel*(-1);
//            m_vel.m_x -= 0.02;
//        }

//    }
    //m_pos+=m_vel;
    //randomVel test
    //m_vel+=randVel/8; //try *0.25 (faster)


    m_vel+=m_acc;

    // if statement for limiting velocity by max speed
    auto NormV = m_vel;
    auto speed = m_vel.length();
    if (speed > max_speed)
    {
        NormV.normalize();
        m_vel = NormV * max_speed;
    }

    m_pos+=m_vel;
    m_acc*=0;

    seek(ngl::Vec3(10.0f,0.0f,0.0f)); //seek target
    //seek(randVel);
    updateRotation();
    m_hit=false;
}
