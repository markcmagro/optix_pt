#pragma once

#include <sutil/vec_math.h>

//
// Barycentric coordinates
//

// References:
// - https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf.

inline __device__ float calcTriArea2D(float x1, float y1, float x2, float y2, float x3, float y3)
{
    return (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);
}

inline __device__ void calcBarycentric(float3 a, float3 b, float3 c, float3 p, float &u, float &v, float &w)
{
    float3 m = cross(b - a, c - a);
    float nu, nv, ood;
    float x = fabs(m.x);
    float y = fabs(m.y);
    float z = fabs(m.z);

    if (x >= y && x >= z) {
        nu = calcTriArea2D(p.y, p.z, b.y, b.z, c.y, c.z);
        nv = calcTriArea2D(p.y, p.z, c.y, c.z, a.y, a.z);
        ood = 1.0f / m.x;
    }
    else if (y >= x && y >= z) {
        nu = calcTriArea2D(p.x, p.z, b.x, b.z, c.x, c.z);
        nv = calcTriArea2D(p.x, p.z, c.x, c.z, a.x, a.z);
        ood = 1.0f / -m.y;
    }
    else {
        nu = calcTriArea2D(p.x, p.y, b.x, b.y, c.x, c.y);
        nv = calcTriArea2D(p.x, p.y, c.x, c.y, a.x, a.y);
        ood = 1.0f / m.z;
    }

    u = nu * ood;
    v = nv * ood;
    w = 1.0f - u - v;
}
