#ifndef BOID_H
#define BOID_H

#include <vector>
#include <ngl/Vec3.h>
#include <ngl/Vec4.h>
#include <ngl/Mat4.h>
#include <ngl/ShaderLib.h>
#include <ngl/Transformation.h>
#include <ngl/VAOPrimitives.h>

class Flock;

class Boid
{
public:
    Boid(ngl::Vec3 _pos, ngl::Vec3 _vel, Flock *_flock);
    void draw(const std::string &_shaderName,const ngl::Mat4 &_globalMat, const  ngl::Mat4 &_view, const ngl::Mat4 &_project)const ;
    void loadMatricesToShader(ngl::Transformation &_tx, const ngl::Mat4 &_globalMat, const ngl::Mat4 &_view , const ngl::Mat4 &_project)const;
    void updateRotation();
    void move();

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief setHit function sets the hit (used for BBox collision)
    /// @param m_hit bool variable which sets "hit" to true
    //----------------------------------------------------------------------------------------------------------------------
    inline void setHit(){m_hit=true;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief setNotHit function sets the hit (used for BBox collision)
    /// @param m_hit bool variable which unsets "hit" to false
    //----------------------------------------------------------------------------------------------------------------------
    inline void setNotHit(){m_hit=false;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief isHit function sets the hit (used for BBox collision)
    /// @param m_hit variable returns the current state of hit
    //----------------------------------------------------------------------------------------------------------------------
    inline bool isHit()const {return m_hit;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief getPos function gets the position
    /// @param m_pos position
    //----------------------------------------------------------------------------------------------------------------------
    inline ngl::Vec3 getPos() const {return m_pos;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief getRadius function gets the "radius" (used for BBox collision)
    /// @param m_radius radius (prevents from collision)
    //----------------------------------------------------------------------------------------------------------------------
    inline GLfloat getRadius() const {return m_radius;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief setVel function sets the velocity
    /// @param m_vel velocity
    //----------------------------------------------------------------------------------------------------------------------
    inline void setVel(ngl::Vec3 _v){m_vel=_v;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief getVel function gets the velocity
    /// @param m_vel velocity
    //----------------------------------------------------------------------------------------------------------------------
    inline ngl::Vec3 getVel() const { return m_vel;}
    //----------------------------------------------------------------------------------------------------------------------

    void applyForce(ngl::Vec3 _force);
    void seek(ngl::Vec3 _target);

    bool m_wire;

    ngl::Vec3 m_force;
    ngl::Vec3 m_target;
    ngl::Vec3 m_steer;

private:
    ngl::Vec3 m_pos;
    ngl::Vec3 m_vel;
    ngl::Vec3 m_rotation;
    //ngl::Vec3 m_target;
    GLfloat m_radius;

    ngl::Vec3 m_acc;
    float max_speed;
    float max_force;
    ngl::Vec3 m_desired;

    bool m_hit;
    const Flock *m_flock;
};

#endif // BOID_H
