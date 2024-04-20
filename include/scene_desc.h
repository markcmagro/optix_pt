#pragma once

#include <string>
#include <sutil/Matrix.h>
#include <vector>
#include "light.h"
#include "rapidjson/document.h"

class SceneDesc
{
public:
    const char *gltfSceneFile;
    float3 gltfUpHint;

    int resX;
    int resY;

    float3 bgColour;

    float3 camOrigin;
    float3 camUp;
    float3 camTarget;
    float camFovYDeg;
    float camAspectRatio;

    std::vector<Light> lights;

    std::string sceneStr;
    rapidjson::Document doc;

    SceneDesc(const char *jsonSceneFile);
    ~SceneDesc();

    virtual bool parse();

private:
    bool parseGltfScene();
    bool parseRenderer();
};

bool getSceneFilename(const std::string &sceneListFilename, std::string &sceneFilename);
void printJsonFragment(std::string &jsonStr, size_t offset);
