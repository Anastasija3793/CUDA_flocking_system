#include <QCoreApplication>
#include <benchmark/benchmark.h>
#include <chrono>

#include "Flock.h"
#include "libFlockGPU.h"

//----------------------------------10---BOIDS------------------------------------------------------------------------
static void FlockCPU_30_Update(benchmark::State& state)
{
    Flock FlockCPU(30);

    for(auto _ : state)
    {
        FlockCPU.update();
    }
}
BENCHMARK(FlockCPU_30_Update);

static void FlockGPU_30_Update(benchmark::State& state)
{
    libFlockGPU FlockGPU(30);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.update();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_30_Update)->UseManualTime();



static void FlockCPU_30_Flock(benchmark::State& state)
{
    Flock FlockCPU(30);

    for(auto _ : state)
    {
        FlockCPU.flock();
    }
}
BENCHMARK(FlockCPU_30_Flock);

static void FlockGPU_30_Flock(benchmark::State& state)
{
    libFlockGPU FlockGPU(30);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.flock();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_30_Flock)->UseManualTime();



static void FlockCPU_30_Separate(benchmark::State &state)
{
    Flock FlockCPU(30);

    for(auto _ : state)
    {
        FlockCPU.separate();
    }
}
BENCHMARK(FlockCPU_30_Separate);

static void FlockGPU_30_Separate(benchmark::State& state)
{
    libFlockGPU FlockGPU(30);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.separate();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_30_Separate)->UseManualTime();



static void FlockCPU_30_Align(benchmark::State &state)
{
    Flock FlockCPU(30);

    for(auto _ : state)
    {
        FlockCPU.align();
    }
}
BENCHMARK(FlockCPU_30_Align);

static void FlockGPU_30_Align(benchmark::State& state)
{
    libFlockGPU FlockGPU(30);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.align();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_30_Align)->UseManualTime();



static void FlockCPU_30_Cohesion(benchmark::State &state)
{
    Flock FlockCPU(30);

    for(auto _ : state)
    {
        FlockCPU.cohesion();
    }
}
BENCHMARK(FlockCPU_30_Cohesion);

static void FlockGPU_30_Cohesion(benchmark::State& state)
{
    libFlockGPU FlockGPU(30);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.cohesion();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_30_Cohesion)->UseManualTime();

//----------------------------------10---BOIDS------------------------------------------------------------------------




//----------------------------------100---BOIDS------------------------------------------------------------------------
static void FlockCPU_100_Update(benchmark::State& state)
{
    Flock FlockCPU(100);

    for(auto _ : state)
    {
        FlockCPU.update();
    }
}
BENCHMARK(FlockCPU_100_Update);

static void FlockGPU_100_Update(benchmark::State& state)
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
BENCHMARK(FlockGPU_100_Update)->UseManualTime();



static void FlockCPU_100_Flock(benchmark::State& state)
{
    Flock FlockCPU(100);

    for(auto _ : state)
    {
        FlockCPU.flock();
    }
}
BENCHMARK(FlockCPU_100_Flock);

static void FlockGPU_100_Flock(benchmark::State& state)
{
    libFlockGPU FlockGPU(100);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.flock();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_100_Flock)->UseManualTime();



static void FlockCPU_100_Separate(benchmark::State &state)
{
    Flock FlockCPU(100);

    for(auto _ : state)
    {
        FlockCPU.separate();
    }
}
BENCHMARK(FlockCPU_100_Separate);

static void FlockGPU_100_Separate(benchmark::State& state)
{
    libFlockGPU FlockGPU(100);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.separate();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_100_Separate)->UseManualTime();



static void FlockCPU_100_Align(benchmark::State &state)
{
    Flock FlockCPU(100);

    for(auto _ : state)
    {
        FlockCPU.align();
    }
}
BENCHMARK(FlockCPU_100_Align);

static void FlockGPU_100_Align(benchmark::State& state)
{
    libFlockGPU FlockGPU(100);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.align();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_100_Align)->UseManualTime();



static void FlockCPU_100_Cohesion(benchmark::State &state)
{
    Flock FlockCPU(100);

    for(auto _ : state)
    {
        FlockCPU.cohesion();
    }
}
BENCHMARK(FlockCPU_100_Cohesion);

static void FlockGPU_100_Cohesion(benchmark::State& state)
{
    libFlockGPU FlockGPU(100);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.cohesion();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_100_Cohesion)->UseManualTime();

//static void FlockGPU_Update_100(benchmark::State& state)
//{
//    libFlockGPU FlockGPU(100);

//    for(auto _ : state)
//    {
//        FlockGPU.update();
//    }
//}
//BENCHMARK(FlockGPU_Update_100);

//----------------------------------100---BOIDS------------------------------------------------------------------------




//----------------------------------10000---BOIDS------------------------------------------------------------------------
static void FlockCPU_10000_Update(benchmark::State& state)
{
    Flock FlockCPU(10000);

    for(auto _ : state)
    {
        FlockCPU.update();
    }
}
BENCHMARK(FlockCPU_10000_Update);

static void FlockGPU_10000_Update(benchmark::State& state)
{
    libFlockGPU FlockGPU(10000);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.update();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_10000_Update)->UseManualTime();



static void FlockCPU_10000_Flock(benchmark::State& state)
{
    Flock FlockCPU(10000);

    for(auto _ : state)
    {
        FlockCPU.flock();
    }
}
BENCHMARK(FlockCPU_10000_Flock);

static void FlockGPU_10000_Flock(benchmark::State& state)
{
    libFlockGPU FlockGPU(10000);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.flock();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_10000_Flock)->UseManualTime();



static void FlockCPU_10000_Separate(benchmark::State &state)
{
    Flock FlockCPU(10000);

    for(auto _ : state)
    {
        FlockCPU.separate();
    }
}
BENCHMARK(FlockCPU_10000_Separate);

static void FlockGPU_10000_Separate(benchmark::State& state)
{
    libFlockGPU FlockGPU(10000);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.separate();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_10000_Separate)->UseManualTime();



static void FlockCPU_10000_Align(benchmark::State &state)
{
    Flock FlockCPU(10000);

    for(auto _ : state)
    {
        FlockCPU.align();
    }
}
BENCHMARK(FlockCPU_10000_Align);

static void FlockGPU_10000_Align(benchmark::State& state)
{
    libFlockGPU FlockGPU(10000);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.align();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_10000_Align)->UseManualTime();



static void FlockCPU_10000_Cohesion(benchmark::State &state)
{
    Flock FlockCPU(10000);

    for(auto _ : state)
    {
        FlockCPU.cohesion();
    }
}
BENCHMARK(FlockCPU_10000_Cohesion);

static void FlockGPU_10000_Cohesion(benchmark::State& state)
{
    libFlockGPU FlockGPU(10000);

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        FlockGPU.cohesion();
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(FlockGPU_10000_Cohesion)->UseManualTime();

//----------------------------------10000---BOIDS------------------------------------------------------------------------




BENCHMARK_MAIN();
