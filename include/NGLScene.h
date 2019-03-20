#ifndef NGLSCENE_H_
#define NGLSCENE_H_
#include <ngl/BBox.h>
#include <ngl/Text.h>
#include "WindowParams.h"
#include "Sphere.h"
#include <QOpenGLWindow>
#include <memory>

#include "Boid.h"
#include "Flock.h"
//----------------------------------------------------------------------------------------------------------------------
/// @file NGLScene.h
/// @brief this class inherits from the Qt OpenGLWindow and allows us to use NGL to draw OpenGL
/// @author Jonathan Macey
/// @version 1.0
/// @date 10/9/13
/// Revision History :
/// This is an initial version used for the new NGL6 / Qt 5 demos
/// @class NGLScene
/// @brief our main glwindow widget for NGL applications all drawing elements are
/// put in this file
//----------------------------------------------------------------------------------------------------------------------

class NGLScene : public QOpenGLWindow
{
  public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief ctor for our NGL drawing class
    /// @param [in] parent the parent window to the class
    //----------------------------------------------------------------------------------------------------------------------
    NGLScene(/*int _numSpheres*/);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief dtor must close down ngl and release OpenGL resources
    //----------------------------------------------------------------------------------------------------------------------
    ~NGLScene() override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the initialize class is called once when the window is created and we have a valid GL context
    /// use this to setup any default GL stuff
    //----------------------------------------------------------------------------------------------------------------------
    void initializeGL() override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we want to draw the scene
    //----------------------------------------------------------------------------------------------------------------------
    void paintGL() override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we resize
    //----------------------------------------------------------------------------------------------------------------------
    void resizeGL(int _w, int _h) override;

private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the windows params such as mouse and rotations etc
    //----------------------------------------------------------------------------------------------------------------------
    WinParams m_win;

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief used to store the global mouse transforms
    //----------------------------------------------------------------------------------------------------------------------
    ngl::Mat4 m_mouseGlobalTX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our Camera
    //----------------------------------------------------------------------------------------------------------------------
    ngl::Mat4 m_view;
    ngl::Mat4 m_project;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the model position for mouse movement
    //----------------------------------------------------------------------------------------------------------------------
    ngl::Vec3 m_modelPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a dynamic array to contain our spheres
    //----------------------------------------------------------------------------------------------------------------------
    std::vector <Sphere> m_sphereArray;

    unsigned int m_frame = 0;
    unsigned int max_frames = 250;


    //boid array test
    //std::vector <Boid> m_boidArray;
    std::unique_ptr<Flock>m_flock;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the bounding box to contain the spheres
    //----------------------------------------------------------------------------------------------------------------------
    std::unique_ptr<ngl::BBox> m_bbox;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief flag to indicate if we need to do spheresphere checks
    //----------------------------------------------------------------------------------------------------------------------
    bool m_checkSphereSphere;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of spheres we are creating
    //----------------------------------------------------------------------------------------------------------------------
    int m_numSpheres;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief timer to change the sphere position by calling update()
    //----------------------------------------------------------------------------------------------------------------------
    int m_sphereUpdateTimer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief flag to indicate if animation is active or not
    //----------------------------------------------------------------------------------------------------------------------
    bool m_animate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called once per frame to update the sphere positions
    /// and do the collision detection
    //----------------------------------------------------------------------------------------------------------------------
    void updateScene();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief method to load transform matrices to the shader
    //----------------------------------------------------------------------------------------------------------------------
    void loadMatricesToShader();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief method to load transform matrices to the shader
    //----------------------------------------------------------------------------------------------------------------------
    void loadMatricesToColourShader();
     //----------------------------------------------------------------------------------------------------------------------
    /// @brief Qt Event called when a key is pressed
    /// @param [in] _event the Qt event to query for size etc
    //----------------------------------------------------------------------------------------------------------------------
    void keyPressEvent(QKeyEvent *_event) override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called every time a mouse is moved
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mouseMoveEvent (QMouseEvent * _event ) override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse button is pressed
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mousePressEvent ( QMouseEvent *_event) override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse button is released
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mouseReleaseEvent ( QMouseEvent *_event ) override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse wheel is moved
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void wheelEvent( QWheelEvent *_event) override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void timerEvent( QTimerEvent *_event) override;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief check the collisions
    //----------------------------------------------------------------------------------------------------------------------
    void checkCollisions();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief do the actual sphereSphere collisions
    /// @param[in] _pos1 the position of the first sphere
    ///	@param[in] _radius1 the radius of the first sphere
    /// @param[in] _pos2 the position of the second sphere
    ///	@param[in] _radius2 the radius of the second sphere
    //----------------------------------------------------------------------------------------------------------------------
    bool sphereSphereCollision( ngl::Vec3 _pos1, GLfloat _radius1, ngl::Vec3 _pos2, GLfloat _radius2 );
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief check the bounding box collisions
    //----------------------------------------------------------------------------------------------------------------------
    //void BBoxCollision();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief check the sphere collisions
    //----------------------------------------------------------------------------------------------------------------------
    void checkSphereCollisions();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief reset the sphere array
    //----------------------------------------------------------------------------------------------------------------------
    void resetSpheres();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief add a new sphere
    //----------------------------------------------------------------------------------------------------------------------
    void addSphere();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief remove the last sphere added
    //----------------------------------------------------------------------------------------------------------------------
    void removeSphere();


};



#endif
