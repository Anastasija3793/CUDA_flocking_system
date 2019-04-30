#include "Flock.h"
#include <ngl/Random.h>
#include <sstream>
#include <iostream>
#include <fstream>



Flock::Flock(int _numBoids)
{
    for (int i=0; i< _numBoids; ++i)
    {
        m_boids.push_back(this);
    }
    m_numBoids = _numBoids;
}

Flock::~Flock()
{
    //dctor
}

void Flock::separate()
{
    for(int i=0; i<m_numBoids; ++i)
    {
        //ngl::Vec3 sep = m_boids[i].m_sep;
        ngl::Vec3 sep = ngl::Vec3(0.0f,0.0f,0.0f);
        m_boids[i].separate(sep);
        sep*=1.5;
        //applyForce(sep);
        m_boids[i].applyForce(sep);
    }
}

void Flock::align()
{
    for(int i=0; i<m_numBoids; ++i)
    {
        ngl::Vec3 ali = ngl::Vec3(0.0f,0.0f,0.0f);
        m_boids[i].align(ali);
        ali*=0.02;
        //don't need applyForce in order to achieve fireflies effect
        //m_boids[i].applyForce(ali);
    }
}

void Flock::cohesion()
{
    for(int i=0; i<m_numBoids; ++i)
    {
        ngl::Vec3 coh = ngl::Vec3(0.0f,0.0f,0.0f);
        m_boids[i].cohesion(coh);
        coh*=1.0;
        m_boids[i].applyForce(coh);
    }
}

void Flock::flock()
{
    separate();
    align();
    cohesion();
}

void Flock::update()
{
    for(int i=0; i<m_numBoids; ++i)
    {
        m_boids[i].update();
    }
    flock();
}

// write houdini geo file
void Flock::dumpGeo(uint _frameNumber)
{
            char fname[150];

            std::sprintf(fname,"geo/flock_cpu.%03d.geo",++_frameNumber);
            // we will use a stringstream as it may be more efficient
            std::stringstream ss;
            std::ofstream file;
            file.open(fname);
            if (!file.is_open())
            {
                std::cerr << "failed to Open file "<<fname<<'\n';
                exit(EXIT_FAILURE);
            }
            // write header see here http://www.sidefx.com/docs/houdini15.0/io/formats/geo
            ss << "PGEOMETRY V5\n";
            ss << "NPoints " << m_numBoids << " NPrims 1\n";
            ss << "NPointGroups 0 NPrimGroups 1\n";
            // this is hard coded but could be flexible we have 1 attrib which is Colour
            ss << "NPointAttrib 1  NVertexAttrib 0 NPrimAttrib 2 NAttrib 0\n";
            // now write out our point attrib this case Cd for diffuse colour
            ss <<"PointAttrib \n";
            // default the colour to white
            ss <<"Cd 3 float 1 1 1\n";
            // now we write out the particle data in the format
            // x y z 1 (attrib so in this case colour)
            for(unsigned int i=0; i<m_boids.size(); ++i)
            {


                ss<<m_boids[i].getPos().m_x<<" "<<m_boids[i].getPos().m_y<<" "<<m_boids[i].getPos().m_z << " 1 ";
                //ss<<"("<<_boids[i].cellCol.x<<" "<<_boids[i].cellCol.y<<" "<< _boids[i].cellCol.z<<")\n";
                ss<<"("<<std::abs(1)<<" "<<std::abs(1)<<" "<<std::abs(1)<<")\n";
            }

            // now write out the index values
            ss<<"PrimitiveAttrib\n";
            ss<<"generator 1 index 1 location1\n";
            ss<<"dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
            ss<<"Part "<<m_boids.size()<<" ";
            for(size_t i=0; i<m_boids.size(); ++i)
            {
                ss<<i<<" ";
            }
            ss<<" [0	0]\n";
            ss<<"box_object1 unordered\n";
            ss<<"1 1\n";
            ss<<"beginExtra\n";
            ss<<"endExtra\n";
            // dump string stream to disk;
            file<<ss.rdbuf();
            file.close();

}
