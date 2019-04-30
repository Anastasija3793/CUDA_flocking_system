include(../../common.pri)

# This specifies the exe name
TARGET=gpuTests

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

# where to put the .o files
#OBJECTS_DIR=obj
# core Qt Libs to use add more here if needed.
QT+=gui opengl core
# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
	cache()
	DEFINES +=QT5BUILD
}

QMAKE_CXXFLAGS += -std=c++11 -fPIC -Wall -Wextra -pedantic

CUDA_COMPUTE_ARCH=${CUDA_ARCH}
isEmpty(CUDA_COMPUTE_ARCH) {
    message(CUDA_COMPUTE_ARCH environment variable not set - set this to your local CUDA compute capability.)
}

# where to put moc auto generated files
#MOC_DIR=moc
# on a mac we don't create a .app bundle file ( for ease of multiplatform use)
CONFIG-=app_bundle

# Auto include all .cpp files in the project src directory (can specifiy individually if required)
#HEADERS += \
#    $$PWD/src/*.cuh

SOURCES += $$PWD/src/*.cpp


# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include \
                ../../libFlockGPU/include \
                /public/devel/2018/include/gtest
# where our exe is going to live (root of project)
DESTDIR=./
# add the glsl shader files
OTHER_FILES+=README.md
# were are going to default to a console app
CONFIG += console \
            c++11


#LIBS += -lgtest -pthread
LIBS += -L../../libFlockGPU -lFlockGPU
LIBS += -L/public/devel/2018/lib64 -lgtest -lpthread

QMAKE_RPATHDIR += ../../libFlockGPU


CUDA_SOURCES += $$files($$PWD/src/*.cu)

#NGLPATH=$$(NGLDIR)
#isEmpty(NGLPATH){ # note brace must be here
#	message("including $HOME/NGL")
#	include($(HOME)/NGL/UseNGL.pri)
#}
#else{ # note brace must be here
#	message("Using custom NGL location")
#	include($(NGLDIR)/UseNGL.pri)
#}
include(../../cuda_compiler.pri)
