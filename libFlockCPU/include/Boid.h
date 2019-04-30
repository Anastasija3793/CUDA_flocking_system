#ifndef BOID_H
#define BOID_H

#include <vector>
#include <ngl/Vec3.h>
#include <ngl/Vec4.h>
#include <ngl/Mat4.h>
//#include <ngl/ShaderLib.h>
//#include <ngl/Transformation.h>
//#include <ngl/VAOPrimitives.h>
#include <complex>

#include <ngl/BBox.h>

class Flock;

class Boid
{
public:
    Boid(Flock *_flock);
    Boid(ngl::Vec3 _pos, ngl::Vec3 _vel);
    Boid(ngl::Vec3 _pos, ngl::Vec3 _vel, Flock *_flock);
    Boid(ngl::Vec3 _pos, ngl::Vec3 _vel, ngl::Vec3 _target);
    //Boid(ngl::Vec3 _pos, ngl::Vec3 _vel, ngl::Vec3 _acc);

    void update();

    /// @brief getPos function gets the position
    /// @param m_pos position
    //----------------------------------------------------------------------------------------------------------------------
    inline ngl::Vec3 getPos() const {return m_pos;}
    //----------------------------------------------------------------------------------------------------------------------
    //inline ngl::Vec3 getVel() const {return m_vel;}

    void applyForce(ngl::Vec3 _force);

    void seek(ngl::Vec3& _target);
    void separate(ngl::Vec3& _sepVec);
    void align(ngl::Vec3& _aliVec);
    void cohesion(ngl::Vec3& _cohVec);
    //ngl::Vec3 separate();
    //ngl::Vec3 align();
    //ngl::Vec3 cohesion();


    void flock();

//private:

    ngl::Vec3 m_force;
    ngl::Vec3 m_steer;

    ngl::Vec3 m_pos;
    ngl::Vec3 m_vel;
    float max_speed = 1.0f;

    ngl::Vec3 m_rotation;
    ngl::Vec3 m_target;

    ngl::Vec3 m_acc = ngl::Vec3(0.0f,0.0f,0.0f);
    ngl::Vec3 m_sep = ngl::Vec3(0.0f,0.0f,0.0f);

    float max_force = 0.03f;
    ngl::Vec3 m_desired;

    float m_sepRad = 15.0f; //25
    float m_neighbourDist = 30.0f; //50

    Flock *m_flock; //const Flock *m_flock;
};

#endif // BOID_H
