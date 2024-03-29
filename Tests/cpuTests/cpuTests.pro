# This specifies the exe name
TARGET=cpuTests

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
# where to put moc auto generated files
#MOC_DIR=moc
# on a mac we don't create a .app bundle file ( for ease of multiplatform use)
CONFIG-=app_bundle

# Auto include all .cpp files in the project src directory (can specifiy individually if required)
SOURCES+=$$PWD/src/main.cpp


# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include \
            ../../libFlockCPU/include \
            /public/devel/2018/include/gtest
# where our exe is going to live (root of project)
DESTDIR=./
# add the glsl shader files
OTHER_FILES+=README.md
# were are going to default to a console app
CONFIG += console

QMAKE_CXXFLAGS+=-O0


LIBS+= -lgtest -pthread
LIBS += -L../../libFlockCPU -lFlockCPU
LIBS += -L/public/devel/2018/lib64 -lgtest -lpthread

QMAKE_RPATHDIR += ../../libFlockCPU \

NGLPATH=$$(NGLDIR)
isEmpty(NGLPATH){ # note brace must be here
	message("including $HOME/NGL")
	include($(HOME)/NGL/UseNGL.pri)
}
else{ # note brace must be here
	message("Using custom NGL location")
	include($(NGLDIR)/UseNGL.pri)
}
