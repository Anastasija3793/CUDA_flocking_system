#include <QCoreApplication>
#include <benchmark/benchmark.h>
#include <chrono>

#include "Flock.h"
#include "libFlockGPU.h"

//----------------------------------30---BOIDS------------------------------------------------------------------------
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
        FlockGPU.update();
    }
}
BENCHMARK(FlockGPU_30_Update);



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
        FlockGPU.flock();
    }
}
BENCHMARK(FlockGPU_30_Flock);



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
        FlockGPU.separate();
    }
}
BENCHMARK(FlockGPU_30_Separate);



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
        FlockGPU.align();
    }
}
BENCHMARK(FlockGPU_30_Align);



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
        FlockGPU.cohesion();
    }
}
BENCHMARK(FlockGPU_30_Cohesion);

//----------------------------------30---BOIDS------------------------------------------------------------------------




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
        FlockGPU.update();
    }
}
BENCHMARK(FlockGPU_100_Update);



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
        FlockGPU.flock();
    }
}
BENCHMARK(FlockGPU_100_Flock);



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
        FlockGPU.separate();
    }
}
BENCHMARK(FlockGPU_100_Separate);



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
        FlockGPU.align();
    }
}
BENCHMARK(FlockGPU_100_Align);



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
        FlockGPU.cohesion();
    }
}
BENCHMARK(FlockGPU_100_Cohesion);

//----------------------------------100---BOIDS------------------------------------------------------------------------




//----------------------------------1000---BOIDS------------------------------------------------------------------------
static void FlockCPU_1000_Update(benchmark::State& state)
{
    Flock FlockCPU(1000);

    for(auto _ : state)
    {
        FlockCPU.update();
    }
}
BENCHMARK(FlockCPU_1000_Update);

static void FlockGPU_1000_Update(benchmark::State& state)
{
    libFlockGPU FlockGPU(1000);

    for(auto _ : state)
    {
        FlockGPU.update();
    }
}
BENCHMARK(FlockGPU_1000_Update);



static void FlockCPU_1000_Flock(benchmark::State& state)
{
    Flock FlockCPU(1000);

    for(auto _ : state)
    {
        FlockCPU.flock();
    }
}
BENCHMARK(FlockCPU_1000_Flock);

static void FlockGPU_1000_Flock(benchmark::State& state)
{
    libFlockGPU FlockGPU(1000);

    for(auto _ : state)
    {
        FlockGPU.flock();
    }
}
BENCHMARK(FlockGPU_1000_Flock);



static void FlockCPU_1000_Separate(benchmark::State &state)
{
    Flock FlockCPU(1000);

    for(auto _ : state)
    {
        FlockCPU.separate();
    }
}
BENCHMARK(FlockCPU_1000_Separate);

static void FlockGPU_1000_Separate(benchmark::State& state)
{
    libFlockGPU FlockGPU(1000);

    for(auto _ : state)
    {
        FlockGPU.separate();
    }
}
BENCHMARK(FlockGPU_1000_Separate);



static void FlockCPU_1000_Align(benchmark::State &state)
{
    Flock FlockCPU(1000);

    for(auto _ : state)
    {
        FlockCPU.align();
    }
}
BENCHMARK(FlockCPU_1000_Align);

static void FlockGPU_1000_Align(benchmark::State& state)
{
    libFlockGPU FlockGPU(1000);

    for(auto _ : state)
    {
        FlockGPU.align();
    }
}
BENCHMARK(FlockGPU_1000_Align);



static void FlockCPU_1000_Cohesion(benchmark::State &state)
{
    Flock FlockCPU(1000);

    for(auto _ : state)
    {
        FlockCPU.cohesion();
    }
}
BENCHMARK(FlockCPU_1000_Cohesion);

static void FlockGPU_1000_Cohesion(benchmark::State& state)
{
    libFlockGPU FlockGPU(1000);

    for(auto _ : state)
    {
        FlockGPU.cohesion();
    }
}
BENCHMARK(FlockGPU_1000_Cohesion);

//----------------------------------1000---BOIDS------------------------------------------------------------------------




BENCHMARK_MAIN();
