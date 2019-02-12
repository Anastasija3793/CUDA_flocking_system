#ifndef FLOCK_H
#define FLOCK_H

#include <vector>
#include <ngl/Vec3.h>
#include <ngl/BBox.h>
#include "Boid.h"

class Flock
{
public:
    Flock(ngl::Vec3 _pos, int _numBoids);
    ~Flock();
    void draw(const std::string &_shaderName,const ngl::Mat4 &_globalMat, const  ngl::Mat4 &_view, const ngl::Mat4 &_project)const ;
    void move();

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief BBoxCollision method checks the collision between boids and bbox
    /// and makes sure boids stay insed the bbox (when collide - bounce)
    //----------------------------------------------------------------------------------------------------------------------
    void BBoxCollision();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the bbox
    //----------------------------------------------------------------------------------------------------------------------
    std::unique_ptr<ngl::BBox> m_bbox;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief resetBBox function reseting the bbox in flock/demo
    //----------------------------------------------------------------------------------------------------------------------
    void resetBBox();
    //----------------------------------------------------------------------------------------------------------------------

    int m_numBoids;

private:
    //ngl::Vec3 m_pos;
    std::vector<Boid>m_boids;
    //ngl::Vec3 m_vel;
    //ngl::Vec3 m_target;
};

#endif // FLOCK_H
