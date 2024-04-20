#include <iostream>

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "main_window.h"
#include "scene_desc.h"
#include "utils.h"

void glfwErrorCallback(int error, const char *description);

bool processCommandLineArgs(int argc, char **argv, std::string &sceneListFilename);

int main(int argc, char **argv)
{
	try {
        // Default scene list filename.
        std::string sceneListFilename = "configs/scene_selection.json";

        // Possibly override the default scene list filename.
        if (!processCommandLineArgs(argc, argv, sceneListFilename)) {
            std::cerr << argv[0] << " [<JSON scenes file>]" << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Scene list file is \"" << sceneListFilename << "\".\n";

        std::string sceneFilename;
        if (!getSceneFilename(sceneListFilename, sceneFilename))
            return EXIT_FAILURE;

        std::cout << "Scene file is \"" << sceneFilename << "\".\n";

        SceneDesc sceneDesc(sceneFilename.c_str());
        if (!sceneDesc.parse())
            return 1;

        if (!glfwInit()) {
            std::cerr << "GLFW init failed." << std::endl;
            return 1;
        }

        glfwSetErrorCallback(glfwErrorCallback);

        int glfwMaj, glfwMin, glfwRev;
        glfwGetVersion(&glfwMaj, &glfwMin, &glfwRev);
        std::printf("GLFW version        : %d.%d.%d\n", glfwMaj, glfwMin, glfwRev);
        std::printf("GLFW version string : %s\n", glfwGetVersionString());

        MainWindow *mainWindow = new MainWindow("Main", 100, 100, nullptr, sceneDesc);

        // GLEW requires that an OpenGL context has been created and is current.
        // OpenGL calls can only be made after initialising GLEW.
        GLenum err = glewInit();
        if (err != GLEW_OK) {
            std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
            glfwTerminate();
            return 1;
        }

        mainWindow->initGlfwCallbacks();

        // Setup ImGui.
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(mainWindow->getWindow(), true);
        ImGui_ImplOpenGL3_Init("#version 130");

        const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
        if (mode->width > 1920) {
            ImGuiIO &io = ImGui::GetIO();
            io.Fonts->AddFontFromFileTTF("resources/ProggyClean.ttf", 26.0f);
        }

        mainWindow->init();
        mainWindow->run();

        // Cleanup Imgui.
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        delete mainWindow;
        glfwTerminate();
    }
    catch (const std::runtime_error &e) {
        std::cerr << "Error! " << e.what() << std::endl;
        return 1;
    }

	return 0;
}

void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

bool processCommandLineArgs(int argc, char **argv, std::string &sceneListFilename)
{
    if (argc > 1)
        sceneListFilename = argv[1];
    return true;
}
