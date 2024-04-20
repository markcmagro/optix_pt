#include <glm/glm.hpp>

#include "light.h"

void Light::createPointLight(const float3 &origin, float wattage, const float3 &colour)
{
    type = LightType::PointLight;
    p0 = origin;
    this->colour = colour;
    power = wattage * colour;
    radiance = power / (4.0f * M_PIf); // intensity
}

void Light::createDirectionalLight(const float3 &p0, const float3 &p1, const float3 &p2, const float3 &direction,
    float distance, float wattage, const float3 &colour)
{
    // p0, p1 and p2 define the quad that the light should hit. This quad is
    // needed in renderers that use VPLs, to obtain the initial light path
    // vertex. The quad is unused other renderers, such as path or ray tracers.
    type = LightType::DirectionalLight;
    this->p0 = p0 + distance * -direction;
    this->p1 = p1 + distance * -direction; // anchor
    this->p2 = p2 + distance * -direction;
    /*
    std::printf("Supplied:\n");
    std::printf("p0=(%f, %f, %f) p1=(%f, %f, %f) p2=(%f, %f, %f)\n", p0.x,
        p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    std::printf("Converted:\n");
    std::printf("p0=(%f, %f, %f) p1=(%f, %f, %f) p2=(%f, %f, %f)\n", this->p0.x,
        this->p0.y, this->p0.z, this->p1.x, this->p1.y, this->p1.z, this->p2.x,
        this->p2.y, this->p2.z);
    */
    v1 = p0 - p1;
    v2 = p2 - p1;
    area = length(cross(v1, v2));
    normal = normalize(direction);
    this->distance = distance;
    this->colour = colour;
    power = wattage * colour;
    radiance = power / (4.0f * M_PIf);
}

void Light::createSpotLight(const float3 &origin, const float3 &lookAt, float widthDeg, float falloffStartDeg,
    float wattage, const float3 &colour)
{
    // widthDeg and falloffStartDeg are angles from the cone axis.
    // E.g.:
    //   The biggest possible cone angle is 180 degrees; widthDeg would be 90
    //   degrees and falloffStartDeg would be somewhere between 0 and 90
    //   degrees. Such a cone would have a solid angle of 2 * PI steradians,
    //   equivalent to a hemisphere:
    //   solid angle = 2 * PI * (1 - cos(widthDeg))
    //               = 2 * PI * (1 - cos(90))
    //               = 2 * PI * (1 - 0)
    //               = 2 * PI
    type = LightType::SpotLight;
    this->p0 = origin;
    normal = normalize(lookAt);
    widthRad = glm::radians(widthDeg);
    cosWidth = cosf(widthRad);
    cosFalloffStart = cosf(glm::radians(falloffStartDeg));
    this->colour = colour;
    power = wattage * colour;

    // The solid angle of a cone of width theta is 2 * PI * (1 - cos(theta)).
    // Approximate the spotlight's intensity by using a cone whose width cosine
    // is halfway between cosWidth and cosFalloffStart. See PBR p. 724.

    float cosTheta = 0.5f * (cosWidth + cosFalloffStart);
    float solidAngle = 2.0f * M_PIf * (1.0f - cosTheta);

    radiance = power / solidAngle; // intensity
}
