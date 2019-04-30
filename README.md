# CUDA Fireflies Flocking System

<p align="center">
  <img width="360" height="202" src="Images/gpu.gif">
</p>

## Project Overview

Final year CUDA assignment representing fireflies flocking system. The serialized code is based on [2nd year assignment](https://github.com/Anastasija3793/FlockingSystem) and influenced by Craig Reynolds' work. However, it is modified in order to achieve the behaviour similar to a flock of fireflies.

### Dependencies
- [NGL](https://github.com/NCCA/NGL) (serialized code only)
- [Qt](https://www.qt.io/) (qmake - version 5)
- [Google Benchmark](https://github.com/google/benchmark)
- [Google Test](https://github.com/google/googletest)

#### Dependencies (Device Code)
- [CUDA](https://developer.nvidia.com/cuda-toolkit) (version 9.0 or newer)
- Thrust (shipped with CUDA)

### Project Structure
The project contains the following folders:

- libFlockCPU - library of CPU flocking system implementation (C++)
- libFlockGPU - library of GPU flocking system implementation (CUDA)
- FlockingSystemDemo - demo which creates geo files for both CPU and GPU version
- Benchmark - this project is used to measure the speed-ups
- Tests - tests for checking correctness of CPU and GPU version

## Workflow

The fireflies flocking system is based around three main rules of any flocking system:

- Separation
- Alignment
- Cohesion

Optimizing these rules would lead to speeding up the whole flocking system.

Runing FlockingSystemDemo will allow to export the position of boids as .geo files for both CPU and GPU implementations. These files can be then imported in Houdini for easier visualization.

#### Communication between threads
Communication between threads was achieved by using shared memory.

#### Future Work
The parallel implementation of the flocking system is faster than the serial version. However, there're still some improvements which could be made. The algorithm could be improved, however it wasn't changed in order to present a fair comparison between serial and parallel implementation.

## Results
Benchmarked on:
- CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz x 8
- GPU: NVIDIA GTX 1080

### 30 Boids
|      Benchmark       |      Time      |     CPU       | Iterations |
| ---------------------|----------------|---------------|------------|
| FlockCPU_30_Update   |     97762 ns   |     97762 ns  |       7026 |
| FlockGPU_30_Update   |    165041 ns   |    165043 ns  |       4131 |
| FlockCPU_30_Flock    |    115253 ns   |    115254 ns  |       5999 |
| FlockGPU_30_Flock    |    158048 ns   |    157808 ns  |       4284 |
| FlockCPU_30_Separate |     56329 ns   |     56329 ns  |      12115 |
| FlockGPU_30_Separate |     54815 ns   |     54816 ns  |      12610 |
| FlockCPU_30_Align    |     28546 ns   |     28546 ns  |      24441 |
| FlockGPU_30_Align    |     43522 ns   |     43522 ns  |      16122 |
| FlockCPU_30_Cohesion |     30509 ns   |     30509 ns  |      22974 |
| FlockGPU_30_Cohesion |     58142 ns   |     58131 ns  |      11823 |

### 100 Boids
|      Benchmark        |      Time      |     CPU       | Iterations |
| ----------------------|----------------|---------------|------------|
| FlockCPU_100_Update   |    979532 ns   |    979535 ns  |        667 |
| FlockGPU_100_Update   |    158819 ns   |    158820 ns  |       4357 |
| FlockCPU_100_Flock    |   1226912 ns   |   1226922 ns  |        568 |
| FlockGPU_100_Flock    |    157421 ns   |    157421 ns  |       4463 |
| FlockCPU_100_Separate |    611418 ns   |    611421 ns  |       1128 |
| FlockGPU_100_Separate |     54909 ns   |     54909 ns  |      12435 |
| FlockCPU_100_Align    |    305142 ns   |    305144 ns  |       2292 |
| FlockGPU_100_Align    |     44272 ns   |     44272 ns  |      15710 |
| FlockCPU_100_Cohesion |    311713 ns   |    311715 ns  |       2245 |
| FlockGPU_100_Cohesion |     58363 ns   |     58363 ns  |      11535 |

### 1000 Boids
|      Benchmark         |      Time      |     CPU       | Iterations |
| -----------------------|----------------|---------------|------------|
| FlockCPU_1000_Update   | 120818191 ns   | 120819124 ns  |          6 |
| FlockGPU_1000_Update   |    173802 ns   |    173802 ns  |       4019 |
| FlockCPU_1000_Flock    | 121146074 ns   | 121146591 ns  |          6 |
| FlockGPU_1000_Flock    |    170757 ns   |    170757 ns  |       4085 |
| FlockCPU_1000_Separate |  60716411 ns   |  60716946 ns  |         11 |
| FlockGPU_1000_Separate |     59334 ns   |     59334 ns  |      11468 |
| FlockCPU_1000_Align    |  30146835 ns   |  30146880 ns  |         23 |
| FlockGPU_1000_Align    |     49108 ns   |     49108 ns  |      13634 |
| FlockCPU_1000_Cohesion |  30915406 ns   |  30915659 ns  |         23 |
| FlockGPU_1000_Cohesion |     62177 ns   |     62177 ns  |      10955 |


## References:
- Craig Reynolds (2001). *Boids. Background and Update* [online]. Available from: http://www.red3d.com/cwr/boids/ [Accessed 2019]