// Adapted from https://github.com/boksajak/brdf/blob/master/brdf.h

#pragma once

#include "material_metallic_roughness.h"

inline __device__ float3 baseColorToDiffuseReflectance(float3 baseColor, float metallic)
{
	return baseColor * (1.0f - metallic);
}

// Converts Beckmann roughness (alpha) to Oren-Nayar roughness (sigma).
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de
// Rousiers.
inline __device__ float beckmannAlphaToOrenNayarRoughness(float alpha)
{
	return 0.7071067f * atanf(alpha);
}

// Based on Oren-Nayar's qualitative model.
// Source: "Generalization of Lambert's Reflectance Model" by Oren & Nayar.
inline __device__ float orenNayar(const MaterialMetallicRoughness &mat, const float3 &l, const float3 &v)
{
	// Oren-Nayar roughness (sigma) is in radians - use conversion from Beckmann
    // roughness here.
	float sigma = beckmannAlphaToOrenNayarRoughness(mat.alpha);

    float nDotV = dot(mat.shadingNormal, v);
    float nDotL = dot(mat.shadingNormal, l);

    // Clamp to prevent numerical instability.
    nDotV = fmin(fmax(0.00001f, nDotV), 1.0f);
    nDotL = fmin(fmax(0.00001f, nDotL), 1.0f);

	float thetaV = acosf(nDotV);
	float thetaL = acosf(nDotL);

	float alpha = fmax(thetaV, thetaL);
	float beta = fmin(thetaV, thetaL);

	// Calculate cosine of azimuth angles difference by projecting l and v onto
    // plane defined by mat.shadingNormal. Assume l, v, mat.shadingNormal are
    // normalized.
	float3 lProj = l - nDotL * mat.shadingNormal;
	float3 vProj = v - nDotV * mat.shadingNormal;
	float cosPhiDifference = dot(normalize(vProj), normalize(lProj));

	float sigma2 = sigma * sigma;
	float a = 1.0f - 0.5f * (sigma2 / (sigma2 + 0.33f));
	float b = 0.45f * (sigma2 / (sigma2 + 0.09f));

	return (a + b * fmax(0.0f, cosPhiDifference) * sinf(alpha) * tanf(beta));
}

inline __device__ float3 evalOrenNayar(const MaterialMetallicRoughness &mat, const float3 &l, const float3 &v)
{
    float3 diffuseReflectance = baseColorToDiffuseReflectance(mat.baseColour, mat.metallic);
	return diffuseReflectance * (orenNayar(mat, l, v) * M_1_PIf * dot(mat.shadingNormal, l));
}
