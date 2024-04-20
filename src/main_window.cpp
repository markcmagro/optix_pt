#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "gl_utils.h"
#include "main_window.h"

MainWindow::MainWindow(const char *title, int posX, int posY, GLFWwindow *share, SceneDesc &sceneDesc) :
    GLWindow(title, posX, posY, sceneDesc.resX, sceneDesc.resY, share), width(sceneDesc.resX), height(sceneDesc.resY),
        firstInput(true), lastX(sceneDesc.resX / 2), lastY(sceneDesc.resY / 2), renderer(sceneDesc)
{
    program = 0;
    vao = 0;
    vbo = 0;
    texture = 0;

    modelLoc = -1;
    viewLoc = -1;
    projectionLoc = -1;

    keyA = false;
    keyS = false;
    keyW = false;
    keyD = false;
    key1 = false;
    key2 = false;
    key3 = false;
    movementSpeed = 0.1f;

    leftButtonPressed = false;

    renderer.initOptix();
    renderer.createContext();
    renderer.loadScene();
    renderer.buildAccel();
    renderer.setupCompilationOptions();
    renderer.createModules();
    renderer.createProgramGroups();
    renderer.createPipeline();
    renderer.setupSbt();
    renderer.setupOutputBuffer(width, height);
    renderer.setupCamera(width, height);
    renderer.setupLaunchParams(sceneDesc);
    renderer.createDenoiser();
    renderer.createRandomState();
}

MainWindow::~MainWindow()
{
}

void MainWindow::initGlfwCallbacks()
{
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, MainWindow::onKey);
    glfwSetCursorPosCallback(window, MainWindow::onMouse);
    glfwSetMouseButtonCallback(window, MainWindow::onMouseButton);
    glfwSetScrollCallback(window, MainWindow::onScroll);
}

bool MainWindow::init()
{
    unsigned int vShader = loadShader("shaders/basic.vs", GL_VERTEX_SHADER);
    unsigned int fShader = loadShader("shaders/basic.fs", GL_FRAGMENT_SHADER);
    program = createProgram(vShader, fShader);
    GL(glDeleteShader(vShader));
    GL(glDeleteShader(fShader));

    float vertices[] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 0.0f, 1.0f
    };

    GLint locPos = glGetAttribLocation(program, "pos");
    GLint locTexCoordIn = glGetAttribLocation(program, "texCoordIn");

    GL(glGenVertexArrays(1, &vao));
    GL(glGenBuffers(1, &vbo));

    GL(glBindVertexArray(vao));
    GL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
    GL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));

    GL(glVertexAttribPointer(locPos, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0));
    GL(glEnableVertexAttribArray(locPos));

    GL(glVertexAttribPointer(locTexCoordIn, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float))));
    GL(glEnableVertexAttribArray(locTexCoordIn));

    GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

    GL(glGenTextures(1, &texture));

    GL(glActiveTexture(GL_TEXTURE0)); // optional if there is only one texture
    GL(glBindTexture(GL_TEXTURE_2D, texture));

    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));

    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    GL(glBindTexture(GL_TEXTURE_2D, 0));
    GL(glBindVertexArray(0));

    GL(glUseProgram(program));

    GLint texLoc = glGetUniformLocation(program, "tex");
    GL(glUniform1i(texLoc, 0));

    modelLoc = glGetUniformLocation(program, "model");
    viewLoc = glGetUniformLocation(program, "view");
    projectionLoc = glGetUniformLocation(program, "projection");

    return true;
}

void MainWindow::render()
{
    renderer.updateScene();
    renderer.updateLaunchParams();
    renderer.launch();

    GL(glDisable(GL_DEPTH_TEST));

    GL(glActiveTexture(GL_TEXTURE0));
    GL(glBindTexture(GL_TEXTURE_2D, texture));
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_FLOAT;
    GL(glTexImage2D(GL_TEXTURE_2D, 0, texFormat, width, height, 0, GL_RGBA, texelType, renderer.getFrameBuffer()));

    GL(glBindVertexArray(vao));
    GL(glDrawArrays(GL_TRIANGLES, 0, 6));
    GL(glBindVertexArray(0));
}

void MainWindow::run()
{
    glm::mat4 projection = glm::ortho(0, 1, 1, 0, -1, 1);
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 model = glm::mat4(1.0f);

    GL(glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model)));
    GL(glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view)));
    GL(glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection)));

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        processKeys();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Debug");
        //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
        //    ImGui::GetIO().Framerate);
        ImGui::Text("FPS    : %d", (int)(ImGui::GetIO().Framerate + 0.5f));
        ImGui::Text("Samples: %u", renderer.getSampleCount());
        //ImGui::Text("Camera pos=(%.2f, %.2f, %.2f)", renderer.cam.eye.x, renderer.cam.eye.y, renderer.cam.eye.z);
        //glm::vec3 target = renderer.cam.eye + renderer.cam.lookDir;
        //ImGui::Text("Camera target=(%.2f, %.2f, %.2f)", target.x, target.y, target.z);

        ImGui::NewLine();
        bool denoiserState = renderer.getDenoiserState();
        if (ImGui::Checkbox("Denoising", &denoiserState))
            renderer.setDenoiserState(denoiserState);

        ImGui::NewLine();
        if (ImGui::TreeNode("Visualisation")) {
            static int vis = renderer.getVis();
            ImGui::RadioButton("None", &vis, 0);
            ImGui::RadioButton("Metalness", &vis, 1);
            ImGui::RadioButton("Roughness", &vis, 2);
            ImGui::RadioButton("Occlusion", &vis, 3);
            ImGui::RadioButton("Emission",  &vis, 4);
            ImGui::RadioButton("Shading normals", &vis, 5);
            ImGui::RadioButton("Geometric normals", &vis, 6);
            ImGui::RadioButton("Albedo", &vis, 7);
            ImGui::RadioButton("Glossy colour", &vis, 8);
            ImGui::RadioButton("Specular Colour", &vis, 9);
            ImGui::RadioButton("Sheen Colour", &vis, 10);
            ImGui::RadioButton("Sheen Roughness", &vis, 11);
            ImGui::RadioButton("Alpha", &vis, 12);
            ImGui::NewLine();
            ImGui::TreePop();
            renderer.setVis(vis);
        }

        if (ImGui::TreeNode("Tone mapping")) {
            static bool toneMapping = renderer.getToneMappingState();
            if (ImGui::Checkbox("Enabled##Tone mapping", &toneMapping))
                renderer.setToneMappingState(toneMapping);
            static int toneMappingImpl = renderer.getToneMappingImpl();
            ImGui::NewLine();
            ImGui::Text("Method");
            ImGui::RadioButton("Reinhard", &toneMappingImpl, 0);
            ImGui::RadioButton("ACES (Narkowicz)", &toneMappingImpl, 1);
            ImGui::NewLine();
            ImGui::TreePop();
            renderer.setToneMappingImpl(toneMappingImpl);
        }

        if (ImGui::TreeNode("Gamma correction")) {
            static bool gammaCorrectionState = renderer.getGammaCorrectionState();
            if (ImGui::Checkbox("Enabled##Gamma correction", &gammaCorrectionState))
                renderer.setGammaCorrectionState(gammaCorrectionState);
            static float gammaTerm = renderer.getGammaCorrectionTerm();
            if (ImGui::SliderFloat("##Gamma term", &gammaTerm, 0.1f, 4.0f, "%.2f"))
                renderer.setGammaCorrectionTerm(gammaTerm);
            ImGui::TreePop();
        }

        ImGui::End();

        GL(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
        GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        render();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(getWindow());
    }
}

void MainWindow::processKeys()
{
    bool update = false;
    float moveLeftRight = 0.0f;
    float moveBackwardForward = 0.0f;
    float yaw = 0.0f;   // rotate right/left
    float pitch = 0.0f; // rotate up/down

    if (keyA)
    {
        moveLeftRight = -0.1f * movementSpeed;
        update = true;
    }
    else if (keyD)
    {
        moveLeftRight = 0.1f * movementSpeed;
        update = true;
    }

    if (keyS)
    {
        moveBackwardForward = -0.1f * movementSpeed;
        update = true;
    }
    else if (keyW)
    {
        moveBackwardForward = 0.1f * movementSpeed;
        update = true;
    }

    if (key1) {
        key1 = false;
        renderer.dumpFramebuffer();
    }

    if (key2) {
        key2 = false;
        renderer.dumpAccumFramebufferRgbe();
    }

    if (key3) {
        key3 = false;
        renderer.dumpAccumFramebufferExr();
    }

    if (update)
    {
        renderer.resetSampleCount();
        renderer.cam.incrementPositionAlongDirection(moveLeftRight, 0.0f, moveBackwardForward);
        renderer.cam.setupUvwCoordinatSystem();
    }
}

void MainWindow::onKey(int key, int scancode, int action, int mods)
{
    switch (key)
    {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, 1);
        break;

    case GLFW_KEY_A:
        if (action == GLFW_PRESS)
            keyA = true;
        else if (action == GLFW_RELEASE)
            keyA = false;
        break;

    case GLFW_KEY_S:
        if (action == GLFW_PRESS)
            keyS = true;
        else if (action == GLFW_RELEASE)
            keyS = false;
        break;

    case GLFW_KEY_W:
        if (action == GLFW_PRESS)
            keyW = true;
        else if (action == GLFW_RELEASE)
            keyW = false;
        break;

    case GLFW_KEY_D:
        if (action == GLFW_PRESS)
            keyD = true;
        else if (action == GLFW_RELEASE)
            keyD = false;
        break;

    case GLFW_KEY_1:
        if (action == GLFW_RELEASE)
            key1 = true;
        break;

    case GLFW_KEY_2:
        if (action == GLFW_RELEASE)
            key2 = true;
        break;

    case GLFW_KEY_3:
        if (action == GLFW_RELEASE)
            key3 = true;
        break;

    default:
        break;
    }
}

void MainWindow::onMouse(double xPos, double yPos)
{
    if (checkGuiWantInput())
        return;

    if (firstInput) {
        lastX = xPos;
        lastY = yPos;
        firstInput = false;
    }

    if (leftButtonPressed) {
        float xOffset = (float)(xPos - lastX);
        float yOffset = (float)(yPos - lastY);

        renderer.resetSampleCount();

        // To see what is on the right/left, move the mouse to the right/left.
        renderer.cam.incrementDirection(-glm::radians(xOffset), -glm::radians(yOffset));

        renderer.cam.setupUvwCoordinatSystem();
    }

    lastX = xPos;
    lastY = yPos;
}

void MainWindow::onMouseButton(int button, int action, int mods)
{
    if (checkGuiWantInput())
        return;

    const bool pressed = (action == GLFW_PRESS);
    switch (button)
    {
    case GLFW_MOUSE_BUTTON_LEFT:
        leftButtonPressed = pressed;
        break;
    }

    double x, y;
    glfwGetCursorPos(window, &x, &y);
    lastX = x;
    lastY = y;
}

void MainWindow::onScroll(double xOffset, double yOffset)
{
    if (checkGuiWantInput())
        return;

    if (yOffset == 0.0)
        return;

    if (yOffset > 0.0f)
        movementSpeed *= 2.0f;
    else
        movementSpeed *= 0.5f;
}

void MainWindow::onKey(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    MainWindow *mainWindow = (MainWindow *)glfwGetWindowUserPointer(window);
    mainWindow->onKey(key, scancode, action, mods);
}

void MainWindow::onMouse(GLFWwindow *window, double xPos, double yPos)
{
    MainWindow *mainWindow = (MainWindow *)glfwGetWindowUserPointer(window);
    mainWindow->onMouse(xPos, yPos);
}

void MainWindow::onMouseButton(GLFWwindow *window, int button, int action, int mods)
{
    MainWindow *mainWindow = (MainWindow *)glfwGetWindowUserPointer(window);
    mainWindow->onMouseButton(button, action, mods);
}

void MainWindow::onScroll(GLFWwindow *window, double xOffset, double yOffset)
{
    MainWindow *mainWindow = (MainWindow *)glfwGetWindowUserPointer(window);
    mainWindow->onScroll(xOffset, yOffset);
}

bool MainWindow::checkGuiWantInput()
{
    ImGuiIO &io = ImGui::GetIO();
    return io.WantCaptureMouse;
}
