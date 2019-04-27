/****************************************************************************
basic OpenGL demo modified from http://qt-project.org/doc/qt-5.0/qtgui/openglwindow.html
****************************************************************************/
//#include <QtGui/QGuiApplication>
#include <iostream>
//#include "NGLScene.h"

#include <time.h>
#include <sys/time.h>
#include "Flock.h"


int main()
{
    //std::unique_ptr<Flock>m_flock;
    //m_flock.reset(new Flock(100));
    //Flock m_flock = new Flock(100);

    //if(timer_t)


    struct timeval tim_cpu;
    double t1_cpu, t2_cpu;
    gettimeofday(&tim_cpu, NULL);
    t1_cpu=tim_cpu.tv_sec+(tim_cpu.tv_usec/1000000.0);

    Flock f_cpu(100);

    for(int i = 0; i< 150; i++) //150
    {

        f_cpu.update();

        f_cpu.dumpGeo(i);

    }

    gettimeofday(&tim_cpu, NULL);
    t2_cpu=tim_cpu.tv_sec+(tim_cpu.tv_usec/1000000.0);

     std::cout << "CPU took " << t2_cpu-t1_cpu << "s\n";

     return EXIT_SUCCESS;

}



