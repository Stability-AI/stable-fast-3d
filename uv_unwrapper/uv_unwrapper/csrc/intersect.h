#pragma once

#include "common.h"
#include <vector>

bool triangle_aabb_intersection(const uv_float2 &aabb_min,
                                const uv_float2 &aabb_max, const uv_float2 &v0,
                                const uv_float2 &v1, const uv_float2 &v2);
bool triangle_triangle_intersection(uv_float2 p1, uv_float2 q1, uv_float2 r1,
                                    uv_float2 p2, uv_float2 q2, uv_float2 r2);
