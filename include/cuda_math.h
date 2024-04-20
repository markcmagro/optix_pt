#pragma once

#include <cuComplex.h>

inline __host__ __device__ double cuCnorm(cuDoubleComplex x)
{
    double a = cuCreal(x);
    double b = cuCimag(x);
    return sqrt(a * a + b * b);
}

inline __host__ __device__ cuDoubleComplex cuCsqrt(cuDoubleComplex x)
{
    // References:
    // - https://forums.developer.nvidia.com/t/
    //     additional-cucomplex-functions-cucnorm-cucsqrt-cucexp-and-some-complex-double-functions/36892.

	double radius = cuCabs(x);
	double cosA = x.x / radius;
	cuDoubleComplex out;
	out.x = sqrt(radius * (cosA + 1.0) / 2.0);
	out.y = sqrt(radius * (1.0 - cosA) / 2.0);

	// signbit should be false if x.y is negative
	if (signbit(x.y))
		out.y *= -1.0;

	return out;
}
