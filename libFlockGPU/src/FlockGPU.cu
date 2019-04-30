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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#define PI = 3.14159

//for rand
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)


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

FlockGPU::FlockGPU(int _numBoids)
{
    m_numBoids=_numBoids;

    m_pos.resize(m_numBoids);
    m_dPos.resize(m_numBoids);
    m_dPosX.resize(m_numBoids);
    m_dPosY.resize(m_numBoids);
    m_dPosZ.resize(m_numBoids);

    m_dVel.resize(m_numBoids);
    m_dVelX.resize(m_numBoids);
    m_dVelY.resize(m_numBoids);
    m_dVelZ.resize(m_numBoids);    

    m_dSep.resize(m_numBoids);
    m_dSepX.resize(m_numBoids);
    m_dSepY.resize(m_numBoids);
    m_dSepZ.resize(m_numBoids);

    m_dCoh.resize(m_numBoids);
    m_dCohX.resize(m_numBoids);
    m_dCohY.resize(m_numBoids);
    m_dCohZ.resize(m_numBoids);

    m_dAli.resize(m_numBoids);
    m_dAliX.resize(m_numBoids);
    m_dAliY.resize(m_numBoids);
    m_dAliZ.resize(m_numBoids);

    m_dAcc.resize(m_numBoids);
    m_dAccX.resize(m_numBoids);
    m_dAccY.resize(m_numBoids);
    m_dAccZ.resize(m_numBoids);


    thrust::device_vector <float> rand(NUM_BOIDS*6);
    float * randPtr = thrust::raw_pointer_cast(&rand[0]);
    randFloats(randPtr, NUM_BOIDS*6);


    // start with random pos
    m_dPosX.assign(rand.begin(), rand.begin() + NUM_BOIDS);
    m_dPosY.assign(rand.begin() + NUM_BOIDS, rand.begin() + 2*NUM_BOIDS);
    m_dPosZ.assign(rand.begin() + 2*NUM_BOIDS, rand.begin() + 3*NUM_BOIDS);

    // start with random vel
    m_dVelX.assign(rand.begin() + 3*NUM_BOIDS, rand.begin() + 4*NUM_BOIDS);
    m_dVelY.assign(rand.begin() + 4*NUM_BOIDS, rand.begin() + 5*NUM_BOIDS);
    m_dVelZ.assign(rand.begin() + 5*NUM_BOIDS, rand.begin() + 6*NUM_BOIDS);


    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dPosX.begin(), m_dPosY.begin(), m_dPosZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dPosX.end(),   m_dPosY.end(),   m_dPosZ.end())),
                      m_dPos.begin(),
                      get3dVec());
    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dVelX.begin(), m_dVelY.begin(), m_dVelZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dVelX.end(),   m_dVelY.end(),   m_dVelZ.end())),
                      m_dVel.begin(),
                      get3dVec());

    thrust::fill(m_dAccX.begin(), m_dAccX.begin()+m_numBoids, 0);
    thrust::fill(m_dAccY.begin(), m_dAccY.begin()+m_numBoids, 0);
    thrust::fill(m_dAccZ.begin(), m_dAccZ.begin()+m_numBoids, 0);
    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dAccX.begin(), m_dAccY.begin(), m_dAccZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dAccX.end(),   m_dAccY.end(),   m_dAccZ.end())),
                      m_dAcc.begin(),
                      get3dVec());


    m_dPosPtr = thrust::raw_pointer_cast(&m_dPos[0]);
    m_dVelPtr = thrust::raw_pointer_cast(&m_dVel[0]);

    m_dSepPtr = thrust::raw_pointer_cast(&m_dSep[0]);
    m_dCohPtr = thrust::raw_pointer_cast(&m_dCoh[0]);
    m_dAliPtr = thrust::raw_pointer_cast(&m_dAli[0]);

    m_dAccPtr = thrust::raw_pointer_cast(&m_dAcc[0]);
}

FlockGPU::~FlockGPU()
{

}

void FlockGPU::separate()
{
    unsigned int M = 1024;
    unsigned int N = m_numBoids/M + 1;

    unsigned int blockN = NUM_BOIDS/32 + 1;
    dim3 block2(32, 32); // block of threads (x,y)
    dim3 grid2(blockN, 1); // grid blockN * blockN blocks

    thrust::fill(m_dSepX.begin(), m_dSepX.begin()+m_numBoids, 0);
    thrust::fill(m_dSepY.begin(), m_dSepY.begin()+m_numBoids, 0);
    thrust::fill(m_dSepZ.begin(), m_dSepZ.begin()+m_numBoids, 0);
    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dSepX.begin(), m_dSepY.begin(), m_dSepZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dSepX.end(),   m_dSepY.end(),   m_dSepZ.end())),
                      m_dSep.begin(),
                      get3dVec());

    separateKernel<<<grid2,block2>>>(m_dSepPtr,m_dPosPtr,m_dVelPtr);
    cudaDeviceSynchronize();

    std::vector<float3>m_sep(m_numBoids);
    thrust::copy(m_dSep.begin(),m_dSep.end(),m_sep.begin());
    for(unsigned int i=0; i<m_numBoids; ++i)
    {
        m_sep[i]*=1.5;
    }
    m_dSep = m_sep;

    applyForceKernel<<<N,M>>>(m_dSepPtr,m_dAccPtr);
    cudaDeviceSynchronize();

}

void FlockGPU::align()
{

    unsigned int blockN = NUM_BOIDS/32 + 1;
    dim3 block2(32, 32); // block of threads (x,y)
    dim3 grid2(blockN, 1); // grid blockN * blockN blocks

    thrust::fill(m_dAliX.begin(), m_dAliX.begin()+m_numBoids, 0);
    thrust::fill(m_dAliY.begin(), m_dAliY.begin()+m_numBoids, 0);
    thrust::fill(m_dAliZ.begin(), m_dAliZ.begin()+m_numBoids, 0);
    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dAliX.begin(), m_dAliY.begin(), m_dAliZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dAliX.end(),   m_dAliY.end(),   m_dAliZ.end())),
                      m_dAli.begin(),
                      get3dVec());

    alignmentKernel<<<grid2,block2>>>(m_dAliPtr,m_dPosPtr,m_dVelPtr);
    cudaDeviceSynchronize();

    std::vector<float3>m_ali(m_numBoids);
    thrust::copy(m_dAli.begin(),m_dAli.end(),m_ali.begin());
    for(unsigned int i=0; i<m_numBoids; ++i)
    {
        m_ali[i]*=0.02;
    }
    m_dAli = m_ali;

    cudaDeviceSynchronize();
}

void FlockGPU::cohesion()
{
    unsigned int M = 1024;
    unsigned int N = m_numBoids/M + 1;

    unsigned int blockN = NUM_BOIDS/32 + 1;
    dim3 block2(32, 32); // block of threads (x,y)
    dim3 grid2(blockN, 1); // grid blockN * blockN blocks

    thrust::fill(m_dCohX.begin(), m_dCohX.begin()+m_numBoids, 0);
    thrust::fill(m_dCohY.begin(), m_dCohY.begin()+m_numBoids, 0);
    thrust::fill(m_dCohZ.begin(), m_dCohZ.begin()+m_numBoids, 0);
    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dCohX.begin(), m_dCohY.begin(), m_dCohZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dCohX.end(),   m_dCohY.end(),   m_dCohZ.end())),
                      m_dCoh.begin(),
                      get3dVec());

    cohesionKernel<<<grid2,block2>>>(m_dCohPtr,m_dPosPtr,m_dVelPtr);
    cudaDeviceSynchronize();

    std::vector<float3>m_coh(m_numBoids);
    thrust::copy(m_dCoh.begin(),m_dCoh.end(),m_coh.begin());
    for(unsigned int i=0; i<m_numBoids; ++i)
    {
        m_coh[i]*=1.0;
    }
    m_dCoh = m_coh;

    applyForceKernel<<<N,M>>>(m_dCohPtr,m_dAccPtr);
    cudaDeviceSynchronize();
}

void FlockGPU::flock()
{
    separate();
    align();
    cohesion();
}


void FlockGPU::update()
{
    //N - blocks; M - threads
    unsigned int M = 1024;
    unsigned int N = m_numBoids/M + 1;

    updateKernel<<<N,M>>>(m_dPosPtr,m_dAccPtr,m_dVelPtr);

    flock();

    thrust::copy(m_dPos.begin(),m_dPos.end(),m_pos.begin());

    cudaDeviceSynchronize();
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


        ss<<m_pos[i].x<<" "<<m_pos[i].y<<" "<<m_pos[i].z << " 1 ";
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
