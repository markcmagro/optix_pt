#include "cuda_utils.h"
#include "light.h"
#include "random_number_generators.h"
#include "sampling.h"

inline __device__ void point_sample(Light &light, LCG<8> &random, const float3 &p, LightSample &sample)
{
    sample.position = light.p0;

    // Normally used to compute the cosine of the angle between -lightDir (the
    // negated "to the light" direction vector) and the normal at the light
    // source. For point lights, this value is not applicable and is not used.
    // sample.normal = normalize(p - light.p0);

    sample.pdf = 1.0f;

    sample.lightDir = light.p0 - p;
    float lightDist = length(sample.lightDir);
    sample.lightDir /= lightDist; // normalisation

    sample.radiance = light.radiance / (lightDist * lightDist);

    sample.lightDist = lightDist;
}

inline __device__ void directional_sample(Light &light, LCG<8> &random, const float3 &p, LightSample &sample)
{
    sample.position = p + (-light.normal * light.distance);

    // Normally used to compute the cosine of the angle between -lightDir (the
    // negated "to the light" direction vector) and the normal at the light
    // source. For directional lights, this computation is not required.
    // However, the sample normal is needed to compute the lightDir vector which
    // is used to compute the cosine of the angle between lightDir and the
    // surface normal at the hit point.
    sample.normal = light.normal;

    sample.pdf = 1.0f;

    sample.lightDir = -light.normal;
    sample.radiance = light.radiance;

    sample.lightDist = length(sample.position - p);
}

inline __device__ void spot_sample(Light &light, LCG<8> &random, const float3 &p, LightSample &sample)
{
    sample.position = light.p0;
    sample.normal = light.normal;
    sample.pdf = 1.0f;

    sample.lightDir = light.p0 - p;
    float lightDist = length(sample.lightDir);
    sample.lightDir /= lightDist; // normalisation

    sample.radiance = light.radiance * falloff(light, -sample.lightDir) / (lightDist * lightDist);

    sample.lightDist = lightDist;
}

extern "C" __device__ void __direct_callable__sampleLight(Light &light, LCG<8> &random, const float3 &p,
    LightSample &sample)
{
    switch (light.type)
    {
    case LightType::PointLight:
        point_sample(light, random, p, sample);
        break;

    case LightType::DirectionalLight:
        directional_sample(light, random, p, sample);
        break;

    case LightType::SpotLight:
        spot_sample(light, random, p, sample);
        break;

    default:
        break;
    }
}
