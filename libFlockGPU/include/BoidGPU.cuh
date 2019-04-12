#ifndef BOIDGPU_CUH
#define BOIDGPU_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

class FlockGPU;

class BoidGPU
{
public:
    BoidGPU(/*thrust::device_vector<float> _pos,*/ FlockGPU *_flock);

    //~BoidGPU();

    void update();


    void applyForce(thrust::device_vector<float> _force);
    thrust::device_vector<float> seek(thrust::device_vector<float> _target);
    thrust::device_vector<float> separate();
    thrust::device_vector<float> align();
    thrust::device_vector<float> cohesion();

    void bbox();
    void flock();

private:
    thrust::device_vector<float> m_pos;
    float * m_posPtr;
    thrust::device_vector<float> m_vel;
    float * m_velPtr;

    thrust::device_vector<float> m_steer;
    thrust::device_vector<float> m_target;
    thrust::device_vector<float> m_desired;

    float3 m_force;
    float max_force;
    float max_speed;

    float3 m_acc;
    //float3 m_rotation;
    float m_sepRad = 15.0f; //25
    float m_neighbourDist = 30.0f; //50

    FlockGPU *m_flock;
};

#endif // BOIDGPU_H
