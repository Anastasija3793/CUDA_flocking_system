#include "BoidGPU.cuh"
#include <iostream>

//#define PI = 3.14159

BoidGPU::BoidGPU(/*thrust::device_vector<float> _pos,*/ FlockGPU *_flock)
{
    m_pos.resize(3,0);
    m_vel.resize(3,0);

    m_posPtr = thrust::raw_pointer_cast(&m_pos[0]);
    m_velPtr = thrust::raw_pointer_cast(&m_vel[0]);

    m_flock = _flock;
}

//BoidGPU::~BoidGPU()
//{
    
//}

//void BoidGPU::applyForce(ngl::Vec3 _force)
//{
//    m_acc+=_force;
//}

//void BoidGPU::bbox()
//{
//    thrust::device_vector<float> desiredVel;

//        if(m_pos.m_x >= 2 && m_vel.m_x >0)
//        {
//            desiredVel = thrust::device_vector<float>(-m_vel.m_x,m_vel.m_y,m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox x bounds\n";
//        }
//        else if(m_pos.m_x <= -2 && m_vel.m_x <0)
//        {
//            desiredVel = thrust::device_vector<float>(-m_vel.m_x,m_vel.m_y,m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox -x bounds\n";
//        }
//        else if(m_pos.m_y >= 2 && m_vel.m_y >0)
//        {
//            desiredVel = thrust::device_vector<float>(m_vel.m_x,-m_vel.m_y,m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox y bounds\n";
//        }
//        else if(m_pos.m_y <= -2 && m_vel.m_y <0)
//        {
//            desiredVel = thrust::device_vector<float>(m_vel.m_x,-m_vel.m_y,m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox -y bounds\n";
//        }
//        else if(m_pos.m_z >= 2 && m_vel.m_z >0)
//        {
//            desiredVel = thrust::device_vector<float>(m_vel.m_x,m_vel.m_y,-m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox z bounds\n";
//        }
//        else if(m_pos.m_z <= -2 && m_vel.m_z <0)
//        {
//            desiredVel = thrust::device_vector<float>(m_vel.m_x,m_vel.m_y,-m_vel.m_z);
//            m_vel += seek(desiredVel);

//            //std::cout<<" out of bbox -z bounds\n";
//        }
//}

//void BoidGPU::flock()
//{
//    //3 rules
//    thrust::device_vector<float> sep = separate();
//    thrust::device_vector<float> ali = align();
//    thrust::device_vector<float> coh = cohesion();

//    sep*=1.5;
//    ali*=0.02;
//    coh*=1.0;

//    applyForce(sep);
////    applyForce(ali);
//    applyForce(coh);
//}


//thrust::device_vector<float> BoidGPU::seek(thrust::device_vector<float> _target)
//{
//    m_target = _target;
//    m_desired = m_target - m_pos;
//    // Scale to maximum speed
//    m_desired.normalize();
//    m_desired*=max_speed;

//    m_steer = m_desired - m_vel;
//    // Limit to maximum steering force (limit by max_force)
//    if(m_steer.length() > max_force)
//    {
//        m_steer = (m_steer/m_steer.length())*max_force;
//    }
//    return m_steer;
//}

//thrust::device_vector<float> BoidGPU::separate()
//{
//    m_steer = thrust::device_vector<float>(0.0f,0.0f,0.0f);
//    int count = 0;
//    // For every boid check if too close
//    for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
//    {
//        float d = (m_pos - m_flock->m_boids[i].m_pos).length();
//        if((d>0) && (d<m_sepRad))
//        {
//            thrust::device_vector<float> diff = m_pos - m_flock->m_boids[i].m_pos;
//            diff.normalize();
//            diff/=d;
//            m_steer+=diff;
//            count++;
//        }
//    }
//    // Average
//    if(count>0)
//    {
//        m_steer/=(float(count));
//        //m_steer/=count;
//    }
//    if(m_steer.length()>0)
//    {
//        m_steer.normalize();
//        m_steer*=max_speed;
//        m_steer-=m_vel;
//        //limit by max_force
//        if(m_steer.length() > max_force)
//        {
//            m_steer = (m_steer/m_steer.length())*max_force;
//        }
//    }
//    return m_steer;
//}

//thrust::device_vector<float> BoidGPU::align()
//{
//    thrust::device_vector<float> sum = thrust::device_vector<float>(0.0f,0.0f,0.0f);
//    int count = 0;
//    for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
//    {
//        float d = (m_pos - m_flock->m_boids[i].m_pos).length();
//        if((d>0) && (d<m_neighbourDist))
//        {
//            sum+=m_flock->m_boids[i].m_vel;
//            count++;
//        }
//    }
//    if(count>0)
//    {
//        sum/=(float(count));

//        sum.normalize();
//        sum*=max_speed;
//        m_steer = sum - m_vel;
//        return m_steer;
//    }
//    else
//    {
//        return thrust::device_vector<float>(0.0f,0.0f,0.0f);
//    }
//}

//thrust::device_vector<float> BoidGPU::cohesion()
//{
//    thrust::device_vector<float> sum = thrust::device_vector<float>(0.0f,0.0f,0.0f);
//    int count = 0;
//    for(unsigned int i = 0; i<m_flock->m_boids.size(); ++i)
//    {
//        float d = (m_pos - m_flock->m_boids[i].m_pos).length();
//        if((d>0) && (d<m_neighbourDist))
//        {
//            sum+=m_flock->m_boids[i].m_pos;
//            count++;
//        }
//    }
//    if(count>0)
//    {
//        sum/=count;
//        return seek(sum);
//    }
//    else
//    {
//        return ngl::Vec3(0.0f,0.0f,0.0f);
//    }
//}





// BoidGPU::update()
// {
//     //
// }
