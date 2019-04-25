#include <QCoreApplication>
#include <benchmark/benchmark.h>
#include <chrono>

#include "Flock.h"
#include "libFlockGPU.h"


static void FlockCPU_Update_100(benchmark::State& state)
{
    Flock FlockCPU(100);

    for(auto _ : state)
    {
        FlockCPU.update();
    }
}
BENCHMARK(FlockCPU_Update_100);


//static void FlockGPU_Update_100(benchmark::State& state)
//{
//    libFlockGPU FlockGPU(100);

//    for(auto _ : state)
//    {
//        FlockGPU.update();
//    }
//}
//BENCHMARK(FlockGPU_Update_100);


static void FlockGPU_Update_100(benchmark::State& state)
{
    libFlockGPU FlockGPU(100);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.update();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_Update_100)->UseManualTime();

BENCHMARK_MAIN();
