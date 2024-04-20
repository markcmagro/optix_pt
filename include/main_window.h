#pragma once

#include "gl_window.h"
#include "renderer.h"
#include "scene_desc.h"

class MainWindow : public GLWindow
{
    int width;
    int height;

    bool firstInput;
    double lastX;
    double lastY;

    Renderer renderer;

    GLuint program;
    GLuint vao;
    GLuint vbo;
    GLuint texture;

    GLint modelLoc;
    GLint viewLoc;
    GLint projectionLoc;

    bool keyA;
    bool keyS;
    bool keyW;
    bool keyD;
    bool key1; // output framebuffer (PNG)
    bool key2; // output accum framebuffer (RGBE)
    bool key3; // output accum framebuffer (EXR)
    float movementSpeed;

    bool leftButtonPressed;

public:
    MainWindow(const char *title, int posX, int posY, GLFWwindow *share, SceneDesc &sceneDesc);
    virtual ~MainWindow();

    virtual void initGlfwCallbacks();
    virtual bool init();
    virtual void render();
    virtual void run();

private:
    void processKeys();
    void onKey(int key, int scancode, int action, int mods);
    void onMouse(double xPos, double yPos);
    void onMouseButton(int button, int action, int mods);
    void onScroll(double xOffset, double yOffset);

    static void onKey(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void onMouse(GLFWwindow *window, double xPos, double yPos);
    static void onMouseButton(GLFWwindow *window, int button, int action, int mods);
    static void onScroll(GLFWwindow *window, double xOffset, double yOffset);

    bool checkGuiWantInput();
};
