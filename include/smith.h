#pragma once

//
// Smith
//

inline __device__ float smithG(float hDotL, float magNDotL, float hDotV, float magNDotV, float alpha)
{
    // References:
    // - glTF specs B.3.2.
    // - https://github.com/boksajak/brdf/blob/master/brdf.h.
    //   See Smith_G2_Separable.

    // Return 0 if heaviside(hDotL) or heaviside(hDotV) is zero.
    if (hDotL <= 0.0f || hDotV <= 0.0f)
        return 0.0f;

    float alpha2 = alpha * alpha;

    float term1 = (2.0f * magNDotL) / (magNDotL + sqrtf(alpha2 + (1.0f - alpha2) * magNDotL * magNDotL));
    float term2 = (2.0f * magNDotV) / (magNDotV + sqrtf(alpha2 + (1.0f - alpha2) * magNDotV * magNDotV));
    return term1 * term2;
}

inline __device__ float smithG1GGX(float alpha)
{
    // References:
    // - https://github.com/boksajak/brdf/blob/master/brdf.h.
    //   See Smith_G1_GGX.

	float alpha2 = alpha * alpha;
	return 2.0f / (sqrt((alpha2 + 1.0f) / alpha2) + 1.0f);
}

inline __device__ float smithV(float hDotL, float magNDotL, float hDotV, float magNDotV, float alpha)
{
    // References:
    // - glTF specs B.3.2.

    // smithG / (4.0f * magNDotL * magNdotV)
    // Since all the terms in the denominator cancel out with the terms in the
    // numerator, the computation is simplified (see below).

    // Return 0 if heaviside(hDotL) or heaviside(hDotV) is zero.
    if (hDotL <= 0.0f || hDotV <= 0.0f)
        return 0.0f;

    float alpha2 = alpha * alpha;
    float term1 = magNDotL + sqrtf(alpha2 + (1.0f - alpha2) * magNDotL * magNDotL);
    float term2 = magNDotV + sqrtf(alpha2 + (1.0f - alpha2) * magNDotV * magNDotV);
    return 1.0f / (term1 * term2);
}

inline __device__ float smithVJoint(float hDotL, float magNDotL, float hDotV, float magNDotV, float alpha)
{
    // References:
    // - https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/brdf.glsl

    float alpha2 = alpha * alpha;

    float term1 = magNDotL * sqrtf(alpha2 + (1.0f - alpha2) * magNDotV * magNDotV);
    float term2 = magNDotV * sqrtf(alpha2 + (1.0f - alpha2) * magNDotL * magNDotL);

    float ggx = term1 + term2;
    if (ggx > 0.0f)
        return 0.5f / ggx;

    return 0.0f;
}
