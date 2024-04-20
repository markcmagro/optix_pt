#pragma once

#include <sutil/vec_math.h>

struct OrthonormalBasis
{
    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;

    inline __host__ __device__ OrthonormalBasis(const float3 &normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y =  m_normal.x;
            m_binormal.z =  0;
        }
        else
        {
            m_binormal.x =  0;
            m_binormal.y = -m_normal.z;
            m_binormal.z =  m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    inline __host__ __device__ float3 worldToLocal(const float3 &p) const
    {
        return make_float3(dot(p, m_tangent), dot(p, m_binormal),
            dot(p, m_normal));
    }

    inline __host__ __device__ void localToWorld(float3 &p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }
};
