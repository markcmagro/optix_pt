#include <iostream>
#include "gltf_loader.h"
#include "scene_desc.h"
#include "utils.h"

struct rapidJsonError
{
    int errorCode;
    const char *errorString;
};

// Note: It is assumed that the error codes start from zero and increment by one.
rapidJsonError rapidJsonErrorColl[] =
{
    { rapidjson::kParseErrorNone,                          "No error." },
    { rapidjson::kParseErrorDocumentEmpty,                 "The document is empty." },
    { rapidjson::kParseErrorDocumentRootNotSingular,       "The document root must not be followed by other values." },
    { rapidjson::kParseErrorValueInvalid,                  "Invalid value." },
    { rapidjson::kParseErrorObjectMissName,                "Missing object member name." },
    { rapidjson::kParseErrorObjectMissColon,               "Missing colon after object member name." },
    { rapidjson::kParseErrorObjectMissCommaOrCurlyBracket, "Missing comma or '}' after object member." },
    { rapidjson::kParseErrorArrayMissCommaOrSquareBracket, "Missing comma or ']' after array element." },
    { rapidjson::kParseErrorStringUnicodeEscapeInvalidHex, "Incorrect hex digit after \\u escape in string." },
    { rapidjson::kParseErrorStringUnicodeSurrogateInvalid, "Invalid surrogate pair in string." },
    { rapidjson::kParseErrorStringEscapeInvalid,           "Invalid escape character in string." },
    { rapidjson::kParseErrorStringMissQuotationMark,       "Missing closing quotation mark in string." },
    { rapidjson::kParseErrorStringInvalidEncoding,         "Invalid encoding in string." },
    { rapidjson::kParseErrorNumberTooBig,                  "Number too big to be stored in double." },
    { rapidjson::kParseErrorNumberMissFraction,            "Missing fraction part in number." },
    { rapidjson::kParseErrorNumberMissExponent,            "Missing exponent in number." },
    { rapidjson::kParseErrorTermination,                   "Parsing was terminated." },
    { rapidjson::kParseErrorUnspecificSyntaxError,         "Unspecific syntax error." }
};

SceneDesc::SceneDesc(const char *jsonSceneFile)
{
    gltfSceneFile = nullptr;

    bgColour = make_float3(0.0f, 0.0f, 0.0f);

    sceneStr = readFile(jsonSceneFile);
    if (sceneStr.empty())
        throw -1;
}

SceneDesc::~SceneDesc()
{
}

bool SceneDesc::parse()
{
    rapidjson::ParseResult res = doc.Parse(sceneStr.c_str());
    if (!res) {
        std::cout << "Error parsing scene file.\n";
        std::printf("Offset: %zu; Code: %d\n", res.Offset(), res.Code());
        std::cout << "Fragment:\n";
        printJsonFragment(sceneStr, res.Offset());
        return false;
    }

    assert(doc.IsObject());

    if (!doc.HasMember("scene") || !doc["scene"].IsObject()) {
        std::cerr << "Object \"" << "scene" << "\" not found.\n";
        return false;
    }

    if (!parseGltfScene())
        return false;

    if (!parseRenderer())
        return false;

    return true;
}

bool SceneDesc::parseGltfScene()
{
    rapidjson::Value &docScene = doc["scene"];

    gltfSceneFile = docScene["gltf scene file"].GetString();

    if (docScene.HasMember("gltf up hint")) {
        gltfUpHint.x = docScene["gltf up hint"][0].GetFloat();
        gltfUpHint.y = docScene["gltf up hint"][1].GetFloat();
        gltfUpHint.z = docScene["gltf up hint"][2].GetFloat();
    }
    else {
        gltfUpHint = make_float3(0.0f, 1.0f, 0.0f);
    }

    return true;
}

bool SceneDesc::parseRenderer()
{
    rapidjson::Value &docScene = doc["scene"];

    if (!docScene.HasMember("renderer") || !docScene["renderer"].IsObject()) {
        std::cerr << "Object \"" << "renderer" << "\" not found.\n";
        return false;
    }

    rapidjson::Value &sceneRenderer = docScene["renderer"];

    resX = sceneRenderer["dimensions"][0].GetInt();
    resY = sceneRenderer["dimensions"][1].GetInt();

    if (sceneRenderer.HasMember("background colour")) {
        bgColour.x = sceneRenderer["background colour"][0].GetFloat();
        bgColour.y = sceneRenderer["background colour"][1].GetFloat();
        bgColour.z = sceneRenderer["background colour"][2].GetFloat();
    }

    return true;
}

bool getSceneFilename(const std::string &sceneListFilename, std::string &sceneFilename)
{
    std::string sceneListStr = readFile(sceneListFilename.c_str());
    if (sceneListStr.empty()) {
        std::cerr << "Error reading scene list file." << std::endl;
        return false;
    }

    rapidjson::Document doc;
    rapidjson::ParseResult res = doc.Parse(sceneListStr.c_str());
    if (!res)
    {
        std::cerr << "Error parsing scene list.\n";
        std::cerr << "Offset: " << res.Offset() << "; Code: " << res.Code() <<
            " (" << rapidJsonErrorColl[res.Code()].errorString << ")\n";
        std::cout << "Fragment:\n";
        printJsonFragment(sceneListStr, res.Offset());
        return false;
    }

    if (!doc.IsObject() || !doc.HasMember("scene") || !doc["scene"].IsString()) {
        std::cerr << "  Invalid scene list file format." << std::endl;
        return false;
    }

    const char *sceneId = doc["scene"].GetString();

    if (!doc.HasMember(sceneId) || !doc[sceneId].IsString()) {
        std::cerr << "  Scene with ID " << sceneId << " not found." << std::endl;
        return false;
    }

    std::string pathDir = directoryOfFilePath(sceneListFilename);
    sceneFilename = pathDir + doc[sceneId].GetString() + ".json";
    return true;
}

void printJsonFragment(std::string &jsonStr, size_t offset)
{
    int jsonStrLen = static_cast<int>(jsonStr.size());
    int errorOffset = static_cast<int>(offset);

    int startFragment = 0;
    int endFragment = jsonStrLen - 1;

    int index = std::max(0, errorOffset - 1);
    while (index >= 0) {
        if (jsonStr[index] == '\n') {
            startFragment = index + 1;
            break;
        }
        index--;
    }

    index = errorOffset + 1;
    while (index < jsonStrLen) {
        if (jsonStr[index] == '\n') {
            endFragment = index - 1;
            break;
        }
        index++;
    }

    for (size_t i = startFragment; i <= endFragment; ++i)
        std::cout << jsonStr[i];
    std::cout << '\n';
}
