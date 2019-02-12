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

    bool m_wire;

private:
    ngl::Vec3 m_pos;
    ngl::Vec3 m_vel;
    ngl::Vec3 m_rotation;
    //ngl::Vec3 m_target;

    const Flock *m_flock;
};

#endif // BOID_H
