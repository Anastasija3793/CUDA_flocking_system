#include "FlockGPU.cuh"
#include "BoidGPUKernels.cuh"
#include <cuda.h>
#include <curand.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>
#include "random.cuh"

//for rand
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)


FlockGPU::FlockGPU(int _numBoids)
{
    m_numBoids=_numBoids;

    m_dPosX.resize(m_numBoids);
//    m_dPosY.resize(m_numBoids);
    m_dPosZ.resize(m_numBoids);
    m_dVelX.resize(m_numBoids);
//    m_dVelY.resize(m_numBoids);
    m_dVelZ.resize(m_numBoids);
    //m_dTarget.resize(m_numBoids);
    //m_dSteer.resize(m_numBoids);

    //thrust::device_vector<float>m_dPosX = m_dPos.m_x;
    //makefloat3?
    // check how to access x from float3! (or instead float3 - create float posx, posy, posz?)

    thrust::device_vector<float> randPos(NUM_BOIDS*4);
    float * randPosPtr = thrust::raw_pointer_cast(&randPos[0]);
    randFloats(randPosPtr, NUM_BOIDS*4);


//    m_dPosX.assign(randPos.begin(), randPos.begin() + NUM_BOIDS);
//    m_dPosY.assign(randPos.begin() + NUM_BOIDS, randPos.begin() + 2*NUM_BOIDS);
//    m_dPosZ.assign(randPos.begin() + 2*NUM_BOIDS, randPos.begin() + 3*NUM_BOIDS);

//    m_dVelX.assign(randPos.begin() + 3*NUM_BOIDS, randPos.begin() + 4*NUM_BOIDS);
//    m_dVelY.assign(randPos.begin() + 4*NUM_BOIDS, randPos.begin() + 5*NUM_BOIDS);
//    m_dVelZ.assign(randPos.begin() + 5*NUM_BOIDS, randPos.begin() + 6*NUM_BOIDS);
    m_dPosX.assign(randPos.begin(), randPos.begin() + NUM_BOIDS);
    m_dPosZ.assign(randPos.begin() + NUM_BOIDS, randPos.begin() + 2*NUM_BOIDS);

    m_dVelX.assign(randPos.begin() + 2*NUM_BOIDS, randPos.begin() + 3*NUM_BOIDS);
    m_dVelZ.assign(randPos.begin() + 3*NUM_BOIDS, randPos.begin() + 4*NUM_BOIDS);


    m_dPosXPtr = thrust::raw_pointer_cast(&m_dPosX[0]);
//    m_dPosYPtr = thrust::raw_pointer_cast(&m_dPosY[0]);
    m_dPosZPtr = thrust::raw_pointer_cast(&m_dPosZ[0]);

    m_dVelXPtr = thrust::raw_pointer_cast(&m_dVelX[0]);
//    m_dVelYPtr = thrust::raw_pointer_cast(&m_dVelY[0]);
    m_dVelZPtr = thrust::raw_pointer_cast(&m_dVelZ[0]);


    //need to check!!!
    //start from random places
//    m_dPos.assign(randPos.begin(), randPos.begin() + NUM_BOIDS);
    //start with random velocity
    //m_dVel.assign(randPos.begin() + 2*NUM_BOIDS, randPos.begin() + 3*NUM_BOIDS);
//    m_dVel.assign(randPos.begin() + NUM_BOIDS, randPos.begin() + 2*NUM_BOIDS);
    //give random target (for now)
    //m_dTarget.assign(randPos.begin() + 2*NUM_BOIDS, randPos.begin() + 3*NUM_BOIDS);

    //thrust::fill(m_dPos.begin(), m_dPos.end(), make_float3(randPos));
    //thrust::make_tuple(randPos);

    //point to the device vec
//    m_dPosPtr = thrust::raw_pointer_cast(&m_dPos[0]);
//    m_dVelPtr = thrust::raw_pointer_cast(&m_dVel[0]);
    //m_dTargetPtr = thrust::raw_pointer_cast(&m_dTarget[0]);

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

    //float3 pos = make_float3(m_dPosXPtr,m_dPosYPtr,m_dPosZPtr);
//    float3 pos;
//    pos.x = m_dPosXPtr;

    //steerKernel<<<N,M>>>(m_dPosPtr,m_dVelPtr,m_dTargetPtr,m_dTargetPtr);
    //cudaThreadSynchronize();
    updateKernel<<<N,M>>>(m_dPosXPtr,m_dPosZPtr,m_dVelXPtr,m_dVelZPtr);
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


        ss<<m_dPosX[i]<<" "<<0<<" "<<m_dPosZ[i] << " 1 ";
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
