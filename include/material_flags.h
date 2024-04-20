#pragma once

enum MaterialFlags
{
    Uncategorised   = 0,       //    0
    Metal           = 1 <<  0, //    1
    Dielectric      = 1 <<  1, //    2
    Emission        = 1 <<  2, //    4
    Opaque          = 1 <<  3, //    8
    Transmission    = 1 <<  4, //   16
    Reflection      = 1 <<  5, //   32
    Diffuse         = 1 <<  6, //   64
    Subsurface      = 1 <<  7, //  128
    Coat            = 1 <<  8, //  256
    Sheen           = 1 <<  9, //  512
    Lambertian      = 1 << 10, // 1024
    Specular        = 1 << 11, // 2048
};
