#ifndef FLOCK_H
#define FLOCK_H

/// @file Flock.h
/// @brief Library serialized program for creating a flock which contains boids with their attributes and sets of flocking behaviour rules
/// @author Anastasija Belaka
/// @version N/A
/// @date 30/04/2019 Updated to NCCA Coding standard
/// Revision History : https://github.com/Anastasija3793/CUDA_flocking_system
/// Initial Version 12/02/2019

#include <vector>
#include <ngl/Vec3.h>
#include "Boid.h"

//----------------------------------------------------------------------------------------------------------------------
/// @class Flock "Flock.h"
/// @brief Flock class which contains Flock constructor, boids attributes, behaviour/rules
/// @author Anastasija Belaka
/// @version N/A
/// @date 30/04/2019 Updated to NCCA Coding standard
/// Revision History : See https://github.com/Anastasija3793/CUDA_flocking_system
//----------------------------------------------------------------------------------------------------------------------
class Flock
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Flock constructor with its own values
    /// @param[in] _numBoids the number of boids in the flock
    //----------------------------------------------------------------------------------------------------------------------
    Flock(int _numBoids);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the dctor
    //----------------------------------------------------------------------------------------------------------------------
    ~Flock();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief separate function for applying separation rule to all boids in a flock
    //----------------------------------------------------------------------------------------------------------------------
    void separate();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief align function for applying alignment rule to all boids in a flock
    /// note: in order to achieve a fireflies effect, function applyForce is not applied in there
    //----------------------------------------------------------------------------------------------------------------------
    void align();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief cohesion function for applying cohesion rule to all boids in a flock
    //----------------------------------------------------------------------------------------------------------------------
    void cohesion();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief flock function for applying all 3 rules to all boids in a flock
    //----------------------------------------------------------------------------------------------------------------------
    void flock();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief update function
    //----------------------------------------------------------------------------------------------------------------------
    void update();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief dumpGeo function to write/dump geo into houdini files
    /// @param _frameNumber number of the frame
    //----------------------------------------------------------------------------------------------------------------------
    void dumpGeo(const uint _frameNumber);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief m_numBoids number of boids in the flock
    //----------------------------------------------------------------------------------------------------------------------
    int m_numBoids;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief m_boids the container for the boids
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<Boid>m_boids;
    //----------------------------------------------------------------------------------------------------------------------
};

#endif // FLOCK_H
