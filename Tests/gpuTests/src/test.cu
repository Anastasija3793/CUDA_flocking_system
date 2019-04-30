#include <gtest/gtest.h>

#include <cuda.h>
#include <curand.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>
//#include "random.cuh"


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include "libFlockGPU.h"
#include "FlockGPU.cuh"
#include "BoidGPUKernels.cuh"


typedef thrust::tuple<float,float,float> Float3;

struct get3dVec
{
    __host__ __device__ float3 operator()(Float3 a)
    {
        float x = thrust::get<0>(a);
        float y = thrust::get<1>(a);
        float z = thrust::get<2>(a);

        return make_float3(x,y,z);
    }
};

TEST(FlockGPU, 3dVec)
{
    thrust::device_vector<float3> d_test(3);
    thrust::device_vector<float> d_testX(3);
    thrust::device_vector<float> d_testY(3);
    thrust::device_vector<float> d_testZ(3);
    std::vector<float3> h_test(3);

    thrust::fill(d_testX.begin(), d_testX.end(), 0);
    thrust::fill(d_testY.begin(), d_testY.end(), 1);
    thrust::fill(d_testZ.begin(), d_testZ.end(), 2);
    thrust::transform(thrust::make_zip_iterator(make_tuple(d_testX.begin(), d_testY.begin(), d_testZ.begin())),
                      thrust::make_zip_iterator(make_tuple(d_testX.end(),   d_testY.end(),   d_testZ.end())),
                      d_test.begin(),
                      get3dVec());

    thrust::copy(d_test.begin(),d_test.end(),h_test.begin());
    EXPECT_EQ(h_test[0].x,0);
    EXPECT_EQ(h_test[0].y,1);
    EXPECT_EQ(h_test[0].z,2);

    EXPECT_EQ(h_test[1].x,0);
    EXPECT_EQ(h_test[1].y,1);
    EXPECT_EQ(h_test[1].z,2);

    EXPECT_EQ(h_test[2].x,0);
    EXPECT_EQ(h_test[2].y,1);
    EXPECT_EQ(h_test[2].z,2);
}

TEST(FlockGPU, p)
{
    FlockGPU b(3);

    thrust::fill(b.m_dPosX.begin(), b.m_dPosX.end(), 1.0);
    thrust::fill(b.m_dPosY.begin(), b.m_dPosY.end(), 1.0);
    thrust::fill(b.m_dPosZ.begin(), b.m_dPosZ.end(), 1.0);

    thrust::fill(b.m_dVelX.begin(), b.m_dVelX.end(), 1.0);
    thrust::fill(b.m_dVelY.begin(), b.m_dVelY.end(), 1.0);
    thrust::fill(b.m_dVelZ.begin(), b.m_dVelZ.end(), 1.0);

    thrust::transform(thrust::make_zip_iterator(make_tuple(b.m_dPosX.begin(), b.m_dPosY.begin(), b.m_dPosZ.begin())),
                          thrust::make_zip_iterator(make_tuple(b.m_dPosX.end(),   b.m_dPosY.end(),   b.m_dPosZ.end())),
                          b.m_dPos.begin(),
                          get3dVec());

    thrust::transform(thrust::make_zip_iterator(make_tuple(b.m_dVelX.begin(), b.m_dVelY.begin(), b.m_dVelZ.begin())),
                          thrust::make_zip_iterator(make_tuple(b.m_dVelX.end(),   b.m_dVelY.end(),   b.m_dVelZ.end())),
                          b.m_dVel.begin(),
                          get3dVec());

    std::vector<float3>h_p(3);
    //std::vector<float3>h_v(3);
    thrust::copy(b.m_dPos.begin(),b.m_dPos.end(),h_p.begin());
    //thrust::copy(d_newVel.begin(),d_newVel.end(),h_v.begin());


    thrust::device_vector<float3> d_pos(3);
    thrust::device_vector<float> d_posX(3);
    thrust::device_vector<float> d_posY(3);
    thrust::device_vector<float> d_posZ(3);
    std::vector<float3> h_pos(3);
    //float3 * d_posPtr;

    thrust::device_vector<float3> d_vel(3);
    thrust::device_vector<float> d_velX(3);
    thrust::device_vector<float> d_velY(3);
    thrust::device_vector<float> d_velZ(3);
    std::vector<float3> h_vel(3);

    thrust::fill(d_posX.begin(), d_posX.end(), 1.0);
    thrust::fill(d_posY.begin(), d_posY.end(), 1.0);
    thrust::fill(d_posZ.begin(), d_posZ.end(), 1.0);
    thrust::transform(thrust::make_zip_iterator(make_tuple(d_posX.begin(), d_posY.begin(), d_posZ.begin())),
                      thrust::make_zip_iterator(make_tuple(d_posX.end(),   d_posY.end(),   d_posZ.end())),
                      d_pos.begin(),
                      get3dVec());

    thrust::fill(d_velX.begin(), d_velX.end(), 1.0);
    thrust::fill(d_velY.begin(), d_velY.end(), 1.0);
    thrust::fill(d_velZ.begin(), d_velZ.end(), 1.0);
    thrust::transform(thrust::make_zip_iterator(make_tuple(d_velX.begin(), d_velY.begin(), d_velZ.begin())),
                      thrust::make_zip_iterator(make_tuple(d_velX.end(),   d_velY.end(),   d_velZ.end())),
                      d_vel.begin(),
                      get3dVec());


    //std::vector<float> lengthVel(3);
    for(unsigned i = 0; i<3; i++)
    {
        h_pos[i]+=h_vel[i];
    }

    b.update();
//    EXPECT_EQ(b.m_dPosX[1],1.0);
//    EXPECT_EQ(b.m_dPosY[1],1.0);
//    EXPECT_EQ(b.m_dPosZ[1],1.0);

//    EXPECT_EQ(b.m_dVelX[1],1.0);
//    EXPECT_EQ(b.m_dVelY[1],1.0);
//    EXPECT_EQ(b.m_dVelZ[1],1.0);
    EXPECT_EQ(h_p[1].x,1.0);
    EXPECT_EQ(h_p[1].y,1.0);
    EXPECT_EQ(h_p[1].z,1.0);

    EXPECT_EQ(h_p[1].x,h_pos[1].x);
}

//TEST(FlockGPU, newP)
//{
//    thrust::device_vector<float3> d_newPos(3);
//    thrust::device_vector<float> d_newPosX(3);
//    thrust::device_vector<float> d_newPosY(3);
//    thrust::device_vector<float> d_newPosZ(3);

//    thrust::device_vector<float3> d_newVel(3);
//    thrust::device_vector<float> d_newVelX(3);
//    thrust::device_vector<float> d_newVelY(3);
//    thrust::device_vector<float> d_newVelZ(3);

//    thrust::fill(d_newPosX.begin(), d_newPosX.end(), 1.0);
//    thrust::fill(d_newPosY.begin(), d_newPosY.end(), 1.0);
//    thrust::fill(d_newPosZ.begin(), d_newPosZ.end(), 1.0);

//    thrust::transform(thrust::make_zip_iterator(make_tuple(d_newPosX.begin(), d_newPosY.begin(), d_newPosZ.begin())),
//                      thrust::make_zip_iterator(make_tuple(d_newPosX.end(),   d_newPosY.end(),   d_newPosZ.end())),
//                      d_newPos.begin(),
//                      get3dVec());

//    thrust::fill(d_newVelX.begin(), d_newVelX.end(), 1.0);
//    thrust::fill(d_newVelY.begin(), d_newVelY.end(), 1.0);
//    thrust::fill(d_newVelZ.begin(), d_newVelZ.end(), 1.0);
//    thrust::transform(thrust::make_zip_iterator(make_tuple(d_newVelX.begin(), d_newVelY.begin(), d_newVelZ.begin())),
//                      thrust::make_zip_iterator(make_tuple(d_newVelX.end(),   d_newVelY.end(),   d_newVelZ.end())),
//                      d_newVel.begin(),
//                      get3dVec());

//    std::vector<float3>h_p(3);
//    std::vector<float3>h_v(3);
//    thrust::copy(d_newPos.begin(),d_newPos.end(),h_p.begin());
//    thrust::copy(d_newVel.begin(),d_newVel.end(),h_v.begin());

//    EXPECT_EQ(h_p[1].x, 1.0);

//    FlockGPU f(h_p,h_v,3);

////    EXPECT_EQ(f.h_newP[1].x,1);
//    EXPECT_EQ(f.m_pos[1].x,1);

//}


//TEST(FlockGPU, update)
//{

//    thrust::device_vector<float3> d_pos(3);
//    thrust::device_vector<float> d_posX(3);
//    thrust::device_vector<float> d_posY(3);
//    thrust::device_vector<float> d_posZ(3);
//    std::vector<float3> h_pos(3);
//    //float3 * d_posPtr;

//    thrust::device_vector<float3> d_vel(3);
//    thrust::device_vector<float> d_velX(3);
//    thrust::device_vector<float> d_velY(3);
//    thrust::device_vector<float> d_velZ(3);
//    std::vector<float3> h_vel(3);
//    //float3 * d_velPtr;

//    thrust::device_vector<float3> d_acc(3);
//    thrust::device_vector<float> d_accX(3);
//    thrust::device_vector<float> d_accY(3);
//    thrust::device_vector<float> d_accZ(3);
//    std::vector<float3> h_acc(3);
//    //float3 * d_accPtr;
//    float maxSpeed = 1.0;

//    thrust::fill(d_posX.begin(), d_posX.end(), 1);
//    thrust::fill(d_posY.begin(), d_posY.end(), 1);
//    thrust::fill(d_posZ.begin(), d_posZ.end(), 1);
//    thrust::transform(thrust::make_zip_iterator(make_tuple(d_posX.begin(), d_posY.begin(), d_posZ.begin())),
//                      thrust::make_zip_iterator(make_tuple(d_posX.end(),   d_posY.end(),   d_posZ.end())),
//                      d_pos.begin(),
//                      get3dVec());

//    thrust::fill(d_velX.begin(), d_velX.end(), 1);
//    thrust::fill(d_velY.begin(), d_velY.end(), 1);
//    thrust::fill(d_velZ.begin(), d_velZ.end(), 1);
//    thrust::transform(thrust::make_zip_iterator(make_tuple(d_velX.begin(), d_velY.begin(), d_velZ.begin())),
//                      thrust::make_zip_iterator(make_tuple(d_velX.end(),   d_velY.end(),   d_velZ.end())),
//                      d_vel.begin(),
//                      get3dVec());

//    thrust::fill(d_accX.begin(), d_accX.end(), 1);
//    thrust::fill(d_accY.begin(), d_accY.end(), 1);
//    thrust::fill(d_accZ.begin(), d_accZ.end(), 1);
//    thrust::transform(thrust::make_zip_iterator(make_tuple(d_accX.begin(), d_accY.begin(), d_accZ.begin())),
//                      thrust::make_zip_iterator(make_tuple(d_accX.end(),   d_accY.end(),   d_accZ.end())),
//                      d_acc.begin(),
//                      get3dVec());

//    FlockGPU f(3);
//    //FlockGPU f(d_pos,d_vel,3);

//    thrust::copy(d_pos.begin(),d_pos.end(),h_pos.begin());
//    thrust::copy(d_vel.begin(),d_vel.end(),h_vel.begin());
//    thrust::copy(d_acc.begin(),d_acc.end(),h_acc.begin());

//    std::vector<float> lengthVel(3);
//    for(unsigned i = 0; i<3; i++)
//    {
//        lengthVel[i] = length(h_vel[i]);
//        h_vel[i]+=h_acc[i];

//        if(lengthVel[i] > maxSpeed)
//        {
//            h_vel[i] = (h_vel[i]/lengthVel[i])*maxSpeed;
//        }
//        h_pos[i]+=h_vel[i];
//        h_acc[i]*=maxSpeed;
//    }

//    f.update();

//    EXPECT_EQ(h_pos[2].x,f.m_pos[2].x);
//    EXPECT_EQ(h_pos[2].y,f.m_pos[2].y);
//    EXPECT_EQ(h_pos[2].z,f.m_pos[2].z);
////    EXPECT_EQ(h_pos[3].x,b.m_pos[3].x);

//}

