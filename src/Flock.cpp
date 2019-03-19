#include "Flock.h"
#include <ngl/Random.h>
#include <sstream>
#include <iostream>
#include <fstream>

const static int b_extents=20;

Flock::Flock(int _numBoids)
{
    //m_boids.clear();
    ngl::Random *rand=ngl::Random::instance();

    for (int i=0; i< _numBoids; ++i)
    {
        auto randPos = rand->getRandomPoint(b_extents,b_extents,b_extents);
        m_boids.push_back(Boid(randPos,this));
    }
    m_numBoids = _numBoids;
    m_sepRun = false;
}

Flock::~Flock()
{
    //dctor
}

//----------------------------------------------------------------------------------------------------------------------
void Flock::resetBBox()
{
    m_bbox.reset( new ngl::BBox(ngl::Vec3(),80.0f,80.0f,80.0f));
}
//----------------------------------------------------------------------------------------------------------------------

std::vector<Boid*> Flock::getNeighboursSep(int j)
{
    std::vector<Boid*> neighboursSep;
    auto& thisBoid = m_boids[j];
    //ngl::Vec3 sepVector = ngl::Vec3(0,20,0);
    ngl::Vec3 sum = ngl::Vec3(0,0,0);
    int count = 0;

    //ngl::Vec3 steer = (0,0,0);
    for(int i=0; i<m_numBoids; ++i)
    {
        if (i == j) continue;

        auto d = thisBoid.m_pos - m_boids[i].m_pos;
        // 0.5 is a radius of neighbourhood
        //if ((d.length() > 0) && (d.length() < 1.2f))//1.8
        if (d.length() < 1.2f)//1.8
        {
            ngl::Vec3 diff = thisBoid.m_pos - m_boids[i].m_pos;
            diff.normalize();
            //diff = diff.operator /(d);
            sum+=diff;
            count++;

//            if (count>0)
//            {
//                sum/=count;

//                    //std::abs(m_boids[0].max_speed);
//                    auto steer = m_boids[i].m_steer*sum;
//                    m_boids[i].applyForce(steer);

//            }



            neighboursSep.push_back(&m_boids[i]);
                //sepVector = += dir/
                //thisBoid.m_vel*=-1;
                //m_boids[i].m_vel+=thisBoid.m_pos - m_boids[i].m_pos;
            //m_boids[i].m_vel*=-1;
                //thisBoid.m_steer+=sepVector;
        }
    }
    if (count>0)
    {
        sum/=count;
        for(int k=0; k<m_numBoids; ++k)
        {
            //m_boids[k].m_vel.operator *=(-1.0);
            //std::abs(m_boids[k].max_speed);
            auto steer = m_boids[k].m_steer*sum;
            m_boids[k].applyForce(steer);
        }
    }

    return neighboursSep;
}

void Flock::move()
{
    for(int i=0; i<m_numBoids; ++i)
    {
        //auto neighboursSep = getNeighboursSep(i);
        //m_boids[i].setSeparate(neighboursSep);
//        if (m_sepRun)
//        {
//            m_boids[i].separate();
//        }

        m_boids[i].move();
    }
}


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

//void Flock::separation()
//{
//    for(int i=0; i<m_numBoids; ++i)
//    {
//        m_boids[i].separate();
//    }
//}

void Flock::draw(const std::string &_shaderName, const ngl::Mat4 &_globalMat, const ngl::Mat4 &_view, const ngl::Mat4 &_project) const
{
    for(int i=0; i<m_numBoids; ++i)
    {
        m_boids[i].draw(_shaderName, _globalMat, _view, _project);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void Flock::BBoxCollision()
{
      //create an array of the extents of the bounding box
      float ext[6];
      ext[0]=ext[1]=(m_bbox->height()/2.0f);
      ext[2]=ext[3]=(m_bbox->width()/2.0f);
      ext[4]=ext[5]=(m_bbox->depth()/2.0f);
      // Dot product needs a Vector so we convert The Point Temp into a Vector so we can
      // do a dot product on it
      ngl::Vec3 newP;
      // D is the distance of the Agent from the Plane. If it is less than ext[i] then there is
      // no collision
      GLfloat dist;
      // Loop for each sphere in the vector list
      for(Boid &b : m_boids)
      {
        newP=b.getPos();
        //Now we need to check the Sphere agains all 6 planes of the BBOx
        //If a collision is found we change the dir of the Sphere then Break
        for(int i=0; i<6; ++i)
        {
          //to calculate the distance we take the dotporduct of the Plane Normal
          //with the new point P
          dist=m_bbox->getNormalArray()[i].dot(newP);
          //Now Add the Radius of the sphere to the offsett
          dist+=b.getRadius();
          // If this is greater or equal to the BBox extent /2 then there is a collision
          //So we calculate the Spheres new direction
          if(dist >=ext[i])
          {
            //We use the same calculation as in raytracing to determine the
            // the new direction
            GLfloat x= 2*( b.getVel().dot((m_bbox->getNormalArray()[i])));
            ngl::Vec3 d =m_bbox->getNormalArray()[i]*x;
            b.setVel(b.getVel()-d);
            b.setHit();
            //180 to make it rotate "inside"
            //b.m_angle+=180;
          }//end of hit test
         }//end of each face test
        }//end of for
}
//----------------------------------------------------------------------------------------------------------------------
