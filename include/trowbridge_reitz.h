#pragma once

#include <sutil/vec_math.h>
#include "material_metallic_roughness.h"
#include "orthonormal_basis.h"

// Trowbridge-Reitz (TR) / GGX (Walter)
// GTR  = Generalised Trowbridge-Reitz
// GTR1 = Berry
// GTR2 = GTR with tail shape parameter set to 2 = TR = GGX

inline __device__ float trowbridgeReitzD(float nDotH, float alpha, bool debug)
{
    // References:
    // - glTF specs B.3.2.
    // - https://github.com/boksajak/brdf/blob/master/brdf.h.
    //   See GGX_D.

    // Return 0 if heaviside(nDotH) is zero.
    if (nDotH <= 0.0f)
        return 0.0f;

    float alpha2 = fmax(0.00001f, alpha * alpha);
    float temp = nDotH * nDotH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (M_PIf * temp * temp);
}

inline __device__ float3 trowbridgeReitzGetHalfVector(const MaterialMetallicRoughness &mat, const float3 &n,
    const float3 &v, const float2 xi, bool debug)
{
    // The normal is supplied so that the caller can decide whether to use the
    // shading normal or the geometric normal.

    // References:
    // - https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html
    // - https://agraphicsguynotes.com/posts/sample_microfacet_brdf/
    // - https://schuttejoe.github.io/post/ggximportancesamplingpart1/

    float theta = acosf(sqrtf((1.0f - xi.x) / ((mat.alpha2 - 1.0f) * xi.x + 1.0f)));
    float phi = 2.0f * M_PIf * xi.y;
    float3 h = sphericalToCartesian(theta, phi);

    OrthonormalBasis onb(n);
    onb.localToWorld(h);

    return h;
}

inline __device__ float trowbridgeReitzPdf(const float3 &normal, const float3 &h, const float3 &l, const float3 &v,
    float alpha, bool debug)
{
    // References:
    // - https://github.com/boksajak/brdf/blob/master/brdf.h.
    //   See sampleWalterReflectionPdf.

    float nDotH = fmax(0.00001f, dot(normal, h));
    float D = trowbridgeReitzD(nDotH, alpha, debug);

    float hDotV = dot(h, v);
    float pdf = (D * nDotH) / (4.0f * fabs(hDotV));

    return pdf;
}
