#include "FlockGPU.cuh"
#include "BoidGPUKernels.cuh"
#include "Debug.cuh"

#include <cuda.h>
#include <curand.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>
#include "random.cuh"


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

//for rand
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)

//---------------------VERSION1------------TEST------------START----------------------------
//struct random_float3
//{
//    __host__ __device__ float3 operator()(float3 v)
//    {
//        float x = v.x;
//        float y = v.y;
//        float z = v.z;

//        float randX = x*0.5;
//        float randY = y*0.2;
//        float randZ = z*0.7;
//        return make_float3(randX,randY,randZ);
//    }
//};
////for exporting geo
//struct getX
//{
//    __host__ __device__ float operator()(float3 g)
//    {
//        float x = g.x;
//        return x;
//    }

//};
//struct getY
//{
//    __host__ __device__ float operator()(float3 g)
//    {
//        float y = g.y;
//        return y;
//    }

//};
//struct getZ
//{
//    __host__ __device__ float operator()(float3 g)
//    {
//        float z = g.z;
//        return z;
//    }

//};


//FlockGPU::FlockGPU(int _numBoids)
//{
//    m_numBoids=_numBoids;

//    m_dPos.resize(m_numBoids);
//    m_dVel.resize(m_numBoids);
//    m_dPosX.resize(m_numBoids);
////    m_dPosY.resize(m_numBoids);
////    m_dPosZ.resize(m_numBoids);
////    m_dVelX.resize(m_numBoids);
////    m_dVelY.resize(m_numBoids);
////    m_dVelZ.resize(m_numBoids);
//    //m_dTarget.resize(m_numBoids);
//    //m_dSteer.resize(m_numBoids);

//    //thrust::device_vector<float>m_dPosX = m_dPos.m_x;
//    //makefloat3?
//    // check how to access x from float3! (or instead float3 - create float posx, posy, posz?)

////    thrust::device_vector<float> randPos(NUM_BOIDS);//NUM_BOIDS*6
////    float * randPosPtr = thrust::raw_pointer_cast(&randPos[0]);
////    randFloats(randPosPtr, NUM_BOIDS);


//    thrust::transform(m_dPos.begin(),m_dPos.begin()+NUM_BOIDS,m_dPos.begin(),random_float3());
//    thrust::transform(m_dVel.begin(),m_dVel.begin()+2*NUM_BOIDS,m_dVel.begin(),random_float3());


////    m_dPosPtr = thrust::raw_pointer_cast(&m_dPos[0]);
////    m_dVelPtr = thrust::raw_pointer_cast(&m_dVel[0]);

//    m_dPosPtr = thrust::raw_pointer_cast(m_dPos.data());
//    m_dVelPtr = thrust::raw_pointer_cast(m_dVel.data());


//    thrust::transform(m_dPos.begin(),m_dPos.begin()+3*NUM_BOIDS,getposX.begin(),getX());
//    thrust::transform(m_dPos.begin(),m_dPos.begin()+4*NUM_BOIDS,getposY.begin(),getY());
//    thrust::transform(m_dPos.begin(),m_dPos.begin()+5*NUM_BOIDS,getposZ.begin(),getZ());
////    m_dPosX.assign(randPos.begin(), randPos.begin() + NUM_BOIDS);
////    m_dPosY.assign(randPos.begin() + NUM_BOIDS, randPos.begin() + 2*NUM_BOIDS);
////    m_dPosZ.assign(randPos.begin() + 2*NUM_BOIDS, randPos.begin() + 3*NUM_BOIDS);

////    m_dVelX.assign(randPos.begin() + 3*NUM_BOIDS, randPos.begin() + 4*NUM_BOIDS);
////    m_dVelY.assign(randPos.begin() + 4*NUM_BOIDS, randPos.begin() + 5*NUM_BOIDS);
////    m_dVelZ.assign(randPos.begin() + 5*NUM_BOIDS, randPos.begin() + 6*NUM_BOIDS);

//    //---------------------------TEST------------------------------------------------
////    m_dPosX.assign(randPos.begin(), randPos.begin() + NUM_BOIDS);
////    m_dPosZ.assign(randPos.begin() + NUM_BOIDS, randPos.begin() + 2*NUM_BOIDS);

////    m_dVelX.assign(randPos.begin() + 2*NUM_BOIDS, randPos.begin() + 3*NUM_BOIDS);
////    m_dVelZ.assign(randPos.begin() + 3*NUM_BOIDS, randPos.begin() + 4*NUM_BOIDS);
//    //-------------------------------------------------------------------------------


////    m_dPosXPtr = thrust::raw_pointer_cast(m_dPosX.data());
////    m_dPosYPtr = thrust::raw_pointer_cast(&m_dPosY[0]);
////    m_dPosZPtr = thrust::raw_pointer_cast(&m_dPosZ[0]);

////    m_dVelXPtr = thrust::raw_pointer_cast(&m_dVelX[0]);
////    m_dVelYPtr = thrust::raw_pointer_cast(&m_dVelY[0]);
////    m_dVelZPtr = thrust::raw_pointer_cast(&m_dVelZ[0]);




//    //thrust::make_tuple(randPos);

//}
//---------------------VERSION1------------TEST------------END------------------------------



//3-tuple to store 3d vector type
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

// Return a host vector with random values in the range [0,1)
thrust::host_vector<float> random_vector(const size_t N,
                                         unsigned int seed = thrust::default_random_engine::default_seed)
{
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    thrust::host_vector<float> temp(N);
    for(size_t i = 0; i < N; i++) {
        temp[i] = u01(rng);
    }
    return temp;
}

FlockGPU::FlockGPU(int _numBoids)
{
    m_numBoids=_numBoids;

    m_dPos.resize(m_numBoids);
    m_dPosX.resize(m_numBoids);
    m_dPosY.resize(m_numBoids);
    m_dPosZ.resize(m_numBoids);

    m_dVel.resize(m_numBoids);
    m_dVelX.resize(m_numBoids);
    m_dVelY.resize(m_numBoids);
    m_dVelZ.resize(m_numBoids);

    m_dPosX=random_vector(m_numBoids);
    m_dPosY=random_vector(m_numBoids);
    m_dPosZ=random_vector(m_numBoids);

    m_dVelX=random_vector(m_numBoids);
    m_dVelY=random_vector(m_numBoids);
    m_dVelZ=random_vector(m_numBoids);

//    typedef thrust::device_vector<float>::iterator                     FloatIterator;
//    typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> FloatIteratorTuple;
//    typedef thrust::zip_iterator<FloatIteratorTuple>                   Float3Iterator;

//    Float3Iterator pos_first = thrust::make_zip_iterator(make_tuple(m_dPosX.begin(), m_dPosY.begin(), m_dPosZ.begin()));
//    Float3Iterator pos_last  = thrust::make_zip_iterator(make_tuple(m_dPosX.end(), m_dPosY.end(), m_dPosZ.end()));
//    Float3Iterator vel_first = thrust::make_zip_iterator(make_tuple(m_dVelX.begin(), m_dVelY.begin(), m_dVelZ.begin()));
//    Float3Iterator vel_last  = thrust::make_zip_iterator(make_tuple(m_dVelX.end(), m_dVelY.end(), m_dVelZ.end()));

//    thrust::transform(pos_first, pos_last, m_dPos.begin(), get3dVec());
//    thrust::transform(vel_first, vel_last, m_dVel.begin(), get3dVec());

    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dPosX.begin(), m_dPosY.begin(), m_dPosZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dPosX.end(),   m_dPosY.end(),   m_dPosZ.end())),
                      m_dPos.begin(),
                      get3dVec());
    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dVelX.begin(), m_dVelY.begin(), m_dVelZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dVelX.end(),   m_dVelY.end(),   m_dVelZ.end())),
                      m_dVel.begin(),
                      get3dVec());


//    m_dPosPtr = thrust::raw_pointer_cast(&m_dPos[0]);
//    m_dVelPtr = thrust::raw_pointer_cast(&m_dVel[0]);

    m_dPosPtr = thrust::raw_pointer_cast(m_dPos.data());
    m_dVelPtr = thrust::raw_pointer_cast(m_dVel.data());
}

FlockGPU::~FlockGPU()
{

}

void FlockGPU::update()
{

    //N - blocks; M - threads
    unsigned int M = 1024;
    unsigned int N = m_numBoids/M + 1;

    //thrust::fill(m_dTarget.begin(), m_dTarget.begin()+m_numBoids,0);
    //thrust::fill(m_dPosPtr,m_dPosPtr + m_numBoids,0);

    //float3 pos = make_float3(m_dPosXPtr,m_dPosYPtr,m_dPosZPtr);
//    float3 pos;
//    pos.x = m_dPosXPtr;

    //steerKernel<<<N,M>>>(m_dPosPtr,m_dVelPtr,m_dTargetPtr,m_dTargetPtr);
    //cudaThreadSynchronize();
//    updateKernel<<<N,M>>>(m_dPosXPtr,m_dPosYPtr,m_dPosZPtr,m_dVelXPtr,m_dVelYPtr,m_dVelZPtr);
//    cudaThreadSynchronize();
    updateKernel<<<N,M>>>(m_dPosPtr,m_dVelPtr);
    cudaThreadSynchronize();

}

// From: https://github.com/NCCA/cuda_workshops/blob/master/shared/src/random.cu
/**
 * Fill an array with random floats using the CURAND function.
 * \param devData The chunk of memory you want to fill with floats within the range (0,1]
 * \param n The size of the chunk of data
 */
int FlockGPU::randFloats(float *&devData, const size_t n)
{
    // The generator, used for random numbers
    curandGenerator_t gen;

    // Create pseudo-random number generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Set seed to be the current time (note that calls close together will have same seed!)
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    // Generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    // Cleanup
    CURAND_CALL(curandDestroyGenerator(gen));
    return EXIT_SUCCESS;
}

void FlockGPU::dumpGeo(uint _frameNumber)
{
    char fname[150];

    std::sprintf(fname,"geo/flock_gpu.%03d.geo",++_frameNumber);
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
    for(unsigned int i=0; i<m_numBoids; ++i)
    {


        ss<<m_dPosX[i]<<" "<<m_dPosY[i]<<" "<<m_dPosZ[i] << " 1 ";
        //ss<<"("<<_boids[i].cellCol.x<<" "<<_boids[i].cellCol.y<<" "<< _boids[i].cellCol.z<<")\n";
        ss<<"("<<std::abs(1)<<" "<<std::abs(1)<<" "<<std::abs(1)<<")\n";
    }

    // now write out the index values
    ss<<"PrimitiveAttrib\n";
    ss<<"generator 1 index 1 location1\n";
    ss<<"dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
    ss<<"Part "<<m_numBoids<<" ";
    for(size_t i=0; i<m_numBoids; ++i)
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
