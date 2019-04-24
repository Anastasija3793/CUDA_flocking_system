#include <iostream>

#include <time.h>
#include <sys/time.h>
#include "Flock.h"
#include "libFlockGPU.h"


int main()
{
    // GPU flocking system
        struct timeval tim;
        double t1, t2;
        gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);

        libFlockGPU flockGPU(100);

        for(int i = 0; i< 250; i++) //250
        {
            flockGPU.update();
            flockGPU.dumpGeo(i);
        }

        gettimeofday(&tim, NULL);
        t2=tim.tv_sec+(tim.tv_usec/1000000.0);

    std::cout << "GPU took " << t2-t1 << "s\n";



    // CPU flocking system
    struct timeval tim_cpu;
    double t1_cpu, t2_cpu;
    gettimeofday(&tim_cpu, NULL);
    t1_cpu=tim_cpu.tv_sec+(tim_cpu.tv_usec/1000000.0);

    Flock f_cpu(100);

    for(int i = 0; i< 250; i++) //150
    {

        f_cpu.update();

        f_cpu.dumpGeo(i);

    }

    gettimeofday(&tim_cpu, NULL);
    t2_cpu=tim_cpu.tv_sec+(tim_cpu.tv_usec/1000000.0);

     std::cout << "CPU took " << t2_cpu-t1_cpu << "s\n";



     return EXIT_SUCCESS;
     
}
