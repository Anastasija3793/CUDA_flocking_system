/****************************************************************************
basic OpenGL demo modified from http://qt-project.org/doc/qt-5.0/qtgui/openglwindow.html
****************************************************************************/
//#include <QtGui/QGuiApplication>
#include <iostream>
//#include "NGLScene.h"

#include <time.h>
#include <sys/time.h>
#include "Flock.h"


//int main(int argc, char **argv)
//{
//  QGuiApplication app(argc, argv);
//  // create an OpenGL format specifier
//  QSurfaceFormat format;
//  // set the number of samples for multisampling
//  // will need to enable glEnable(GL_MULTISAMPLE); once we have a context
//  format.setSamples(4);
//  #if defined(__APPLE__)
//    // at present mac osx Mountain Lion only supports GL3.2
//    // the new mavericks will have GL 4.x so can change
//    format.setMajorVersion(4);
//    format.setMinorVersion(1);
//  #else
//    // with luck we have the latest GL version so set to this
//    format.setMajorVersion(4);
//    format.setMinorVersion(3);
//  #endif
//  // now we are going to set to CoreProfile OpenGL so we can't use and old Immediate mode GL
//  format.setProfile(QSurfaceFormat::CoreProfile);
//  // now set the depth buffer to 24 bits
//  format.setDepthBufferSize(24);
//  // now we are going to create our scene window
//  QSurfaceFormat::setDefaultFormat(format);
////  int numSpheres;
////  if(argc ==1)
////  {
////    numSpheres=50;
////  }
////  else
////  {
////    numSpheres=atoi(argv[1]);
////  }
//  //NGLScene window(numSpheres);
//  NGLScene window;
//  // and set the OpenGL format
//  //window.setFormat(format);
//  // we can now query the version to see if it worked
//  std::cout<<"Profile is "<<format.majorVersion()<<" "<<format.minorVersion()<<"\n";
//  // set the window size
//  window.resize(1024, 720);
//  // and finally show
//  window.show();

//  return app.exec();
//}


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



