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

// Return a host vector with random values in the range [0,1)
//thrust::host_vector<float> random_vector(const size_t N,
//                                         unsigned int seed = thrust::default_random_engine::default_seed)
//{
//    thrust::default_random_engine rng(seed);
//    thrust::uniform_real_distribution<float> u01(-1.0f, 1.0f);//0,1
//    thrust::host_vector<float> temp(N);
//    for(size_t i = 0; i < N; i++) {
//        temp[i] = u01(rng);
//    }
//    return temp;
//}

// random generator
struct randGen {
    int maxVal;
    randGen(int max) :
            maxVal(max) {
    }

    int operator()() {
        return rand() % maxVal;
    }
};

FlockGPU::FlockGPU(int _numBoids)/* : m_numBoids(_numBoids), m_dPos(m_numBoids), m_dPosX(m_numBoids), m_dPosY(m_numBoids), m_dPosZ(m_numBoids),
    m_dVel(m_numBoids),
    m_dVelX(m_numBoids),
    m_dVelY(m_numBoids),
    m_dVelZ(m_numBoids),
    m_pos(m_numBoids)*/
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

    m_pos.resize(m_numBoids);
    xTest.resize(m_numBoids);
    yTest.resize(m_numBoids);
    zTest.resize(m_numBoids);

    m_dSep.resize(m_numBoids);
    m_dSepX.resize(m_numBoids);
    m_dSepY.resize(m_numBoids);
    m_dSepZ.resize(m_numBoids);


//    typedef thrust::device_vector<float>::iterator                     FloatIterator;
//    typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> FloatIteratorTuple;
//    typedef thrust::zip_iterator<FloatIteratorTuple>                   Float3Iterator;

//    Float3Iterator pos_first = thrust::make_zip_iterator(make_tuple(m_dPosX.begin(), m_dPosY.begin(), m_dPosZ.begin()));
//    Float3Iterator pos_last  = thrust::make_zip_iterator(make_tuple(m_dPosX.end(), m_dPosY.end(), m_dPosZ.end()));
//    Float3Iterator vel_first = thrust::make_zip_iterator(make_tuple(m_dVelX.begin(), m_dVelY.begin(), m_dVelZ.begin()));
//    Float3Iterator vel_last  = thrust::make_zip_iterator(make_tuple(m_dVelX.end(), m_dVelY.end(), m_dVelZ.end()));

//    thrust::transform(pos_first, pos_last, m_dPos.begin(), get3dVec());
//    thrust::transform(vel_first, vel_last, m_dVel.begin(), get3dVec());

    std::vector<float> randPos(m_numBoids*3);
//    std::generate(randPos.begin(), randPos.end(), []() {
//        return rand() % 100;
//    });
    std::generate(randPos.begin(), randPos.end(),
                randGen(100));//500

    std::vector<float> randVel(m_numBoids*3);
//    std::generate(randVel.begin(), randVel.end(), []() {
//        return rand() % 100;
//    });
    std::generate(randVel.begin(), randVel.end(),
                randGen(100));

    m_dPosX = randPos;
    m_dPosY = randPos;
    m_dPosZ = randPos;

    m_dVelX = randVel;
    m_dVelY = randVel;
    m_dVelZ = randVel;


    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dPosX.begin(), m_dPosY.begin(), m_dPosZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dPosX.end(),   m_dPosY.end(),   m_dPosZ.end())),
                      m_dPos.begin(),
                      get3dVec());
    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dVelX.begin(), m_dVelY.begin(), m_dVelZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dVelX.end(),   m_dVelY.end(),   m_dVelZ.end())),
                      m_dVel.begin(),
                      get3dVec());

    m_dPosPtr = thrust::raw_pointer_cast(&m_dPos[0]);
    m_dVelPtr = thrust::raw_pointer_cast(&m_dVel[0]);

    m_dSepPtr = thrust::raw_pointer_cast(&m_dSep[0]);

//    m_dPosPtr = thrust::raw_pointer_cast(m_dPos.data());
//    m_dVelPtr = thrust::raw_pointer_cast(m_dVel.data());
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

    //steerKernel<<<N,M>>>(m_dPosPtr,m_dVelPtr,m_dTargetPtr,m_dTargetPtr);
    //cudaThreadSynchronize();

    unsigned int blockN = NUM_BOIDS/32 + 1;
    dim3 block2(32, 32); // block of (X,Y) threads
    dim3 grid2(blockN, 1); // grid blockN * blockN blocks

    thrust::fill(m_dSepX.begin(), m_dSepX.begin()+m_numBoids, 0);
    thrust::fill(m_dSepY.begin(), m_dSepY.begin()+m_numBoids, 0);
    thrust::fill(m_dSepZ.begin(), m_dSepZ.begin()+m_numBoids, 0);
    thrust::transform(thrust::make_zip_iterator(make_tuple(m_dSepX.begin(), m_dSepY.begin(), m_dSepZ.begin())),
                      thrust::make_zip_iterator(make_tuple(m_dSepX.end(),   m_dSepY.end(),   m_dSepZ.end())),
                      m_dSep.begin(),
                      get3dVec());

//    thrust::copy(m_dSep.begin(),m_dSep.end(),m_sep.begin());
    //reset
//    thrust::fill(m_dSep.begin(), m_dSep.begin()+m_numBoids, 0);
//    thrust::fill(m_sep[0].x.begin(), m_sep[0].x.begin()+m_numBoids, 0);

    flockKernel<<<grid2,block2>>>(m_dSepPtr,m_dPosPtr,m_dVelPtr);
    cudaThreadSynchronize();

    updateKernel<<<N,M>>>(m_dPosPtr,m_dVelPtr);
    cudaThreadSynchronize();

    thrust::copy(m_dPos.begin(),m_dPos.end(),m_pos.begin());

    //print
//    thrust::copy(m_dPosX.begin(),m_dPosX.end(),xTest.begin());
//    thrust::copy(m_dPosY.begin(),m_dPosY.end(),yTest.begin());
//    thrust::copy(m_dPosZ.begin(),m_dPosZ.end(),zTest.begin());
//    std::cout<<"x: "<<m_pos[0].x<<'\n';
//    std::cout<<"y: "<<m_pos[0].y<<'\n';
//    std::cout<<"z: "<<m_pos[0].z<<'\n';
//    std::cout<<"x: "<<xTest[0]<<'\n';
//    std::cout<<"y: "<<yTest[0]<<'\n';
//    std::cout<<"z: "<<zTest[0]<<'\n';
//    printf("%d \n", m_dPosX[0]);

}

//// From: https://github.com/NCCA/cuda_workshops/blob/master/shared/src/random.cu
///**
// * Fill an array with random floats using the CURAND function.
// * \param devData The chunk of memory you want to fill with floats within the range (0,1]
// * \param n The size of the chunk of data
// */
//int FlockGPU::randFloats(float *&devData, const size_t n)
//{
//    // The generator, used for random numbers
//    curandGenerator_t gen;

//    // Create pseudo-random number generator
//    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

//    // Set seed to be the current time (note that calls close together will have same seed!)
//    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

//    // Generate n floats on device
//    CURAND_CALL(curandGenerateUniform(gen, devData, n));

//    // Cleanup
//    CURAND_CALL(curandDestroyGenerator(gen));
//    return EXIT_SUCCESS;
//}

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
