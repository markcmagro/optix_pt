#pragma once

void testMetallicRoughness();
void testFresnel();
float2 testReflect(float2 n, float2 wi);
void testReflection();
bool testRefract(float3 n, float3 wi, float eta);
void testRefraction();
void testSnellsLaw();
void testStructAlignment();
void testOptixMatrices();

void linear_RGB_to_sRGB01(float r_l, float g_l, float b_l, float *r, float *g, float *b);
void sRGB01_to_sRGB8(float r, float g, float b, uint8_t *r8, uint8_t *g8, uint8_t *b8);
void sRGB8_to_sRGB01(uint8_t r8, uint8_t g8, uint8_t b8, float *r, float *g, float *b);
void sRGB01_to_linear_RGB(float r, float g, float b, float *r_l, float *g_l, float *b_l);
void testColourTransferFunctions();
