#pragma once

// References:
// glTF KHR_materials_sheen
// - https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_sheen.
// glTF Sample Viewer
// - https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/brdf.glsl
// Production-Friendly Microfacet Sheen BRDF (Conty Estevez and Kulla, 2017)
// Enterprise PBR Shading Model
// - https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2022x.md.html
// - https://github.com/DassaultSystemes-Technology/EnterprisePBRShadingModel
// - https://github.com/DassaultSystemes-Technology/EnterprisePBRShadingModel/blob/master/res/Sheen_E.exr
// Physically Based Shading at DreamWorks Animation
// - https://blog.selfshadow.com/publications/s2017-shading-course/dreamworks/s2017_pbs_dreamworks_notes.pdf

inline __device__ float charlieSheenD(float nDotH, float alphaG, float debug)
{
    float invR = 1.0f / alphaG;
    float cos2h = nDotH * nDotH;
    float sin2h = 1.0f - cos2h;
    return (2.0f + invR) * powf(sin2h, invR * 0.5f) * (0.5f * M_1_PIf);
}

inline __device__ float l(float x, float alphaG)
{
    float oneMinusAlphaSq = (1.0f - alphaG) * (1.0f - alphaG);
    float a = lerp(21.5473f, 25.3245f, oneMinusAlphaSq);
    float b = lerp(3.82987f, 3.32435f, oneMinusAlphaSq);
    float c = lerp(0.19823f, 0.16801f, oneMinusAlphaSq);
    float d = lerp(-1.97760f, -1.27393f, oneMinusAlphaSq);
    float e = lerp(-4.32054f, -4.85967f, oneMinusAlphaSq);
    return a / (1.0f + b * powf(x, c)) + d * x + e;
}

inline __device__ float lambdaSheen(float cosTheta, float alphaG)
{
    if (fabs(cosTheta) < 0.5f)
        return expf(l(cosTheta, alphaG));
    else
        return expf(2.0f * l(0.5f, alphaG) - l(1.0f - cosTheta, alphaG));
}

inline __device__ float charlieSheenV(float nDotV, float nDotL, float alphaG)
{
    return 1.0f / ((1.0f + lambdaSheen(nDotV, alphaG) + lambdaSheen(nDotL, alphaG)) * (4.0f * nDotV * nDotL));
}

inline __device__ float AshikminSheenD(float nDotH, float alpha)
{
    float cos2h = nDotH * nDotH;
    float sin2h = 1.0f - cos2h;
    float sin4h = sin2h * sin2h;
    return (sin4h + 4.0f * expf(-cos2h / (sin2h * alpha))) / (M_PIf * (1.0f + 4.0f * alpha) * sin4h);
}

inline __device__ float AshikminSheenV(float nDotV, float nDotL)
{
    return 1.0f / (4.0f * (nDotL + nDotV - nDotL * nDotV));
}

/*
inline __device__ float charlieSheenAlbedoScaling(float vDotN, float lDotN)
{
    // Lookups using E():
    // x = vDotN
    // y = sheenRoughness

    return fmin(1.0f - maxValue(sheenColor) * E(vDotN), 1.0f - maxValue(sheenColor) * E(lDotN));
}
*/
