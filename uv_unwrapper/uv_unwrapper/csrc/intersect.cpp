#include "intersect.h"
#include "bvh.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

bool triangle_aabb_intersection(const uv_float2 &aabbMin,
                                const uv_float2 &aabbMax, const uv_float2 &v0,
                                const uv_float2 &v1, const uv_float2 &v2) {
  // Convert the min and max aabb defintion to left, right, top, bottom
  float l = aabbMin.x;
  float r = aabbMax.x;
  float t = aabbMin.y;
  float b = aabbMax.y;

  int b0 = ((v0.x > l) ? 1 : 0) | ((v0.y > t) ? 2 : 0) | ((v0.x > r) ? 4 : 0) |
           ((v0.y > b) ? 8 : 0);
  if (b0 == 3)
    return true;

  int b1 = ((v1.x > l) ? 1 : 0) | ((v1.y > t) ? 2 : 0) | ((v1.x > r) ? 4 : 0) |
           ((v1.y > b) ? 8 : 0);
  if (b1 == 3)
    return true;

  int b2 = ((v2.x > l) ? 1 : 0) | ((v2.y > t) ? 2 : 0) | ((v2.x > r) ? 4 : 0) |
           ((v2.y > b) ? 8 : 0);
  if (b2 == 3)
    return true;

  float m, c, s;

  int i0 = b0 ^ b1;
  if (i0 != 0) {
    if (v1.x != v0.x) {
      m = (v1.y - v0.y) / (v1.x - v0.x);
      c = v0.y - (m * v0.x);
      if (i0 & 1) {
        s = m * l + c;
        if (s >= t && s <= b)
          return true;
      }
      if (i0 & 2) {
        s = (t - c) / m;
        if (s >= l && s <= r)
          return true;
      }
      if (i0 & 4) {
        s = m * r + c;
        if (s >= t && s <= b)
          return true;
      }
      if (i0 & 8) {
        s = (b - c) / m;
        if (s >= l && s <= r)
          return true;
      }
    } else {
      if (l == v0.x || r == v0.x)
        return true;
      if (v0.x > l && v0.x < r)
        return true;
    }
  }

  int i1 = b1 ^ b2;
  if (i1 != 0) {
    if (v2.x != v1.x) {
      m = (v2.y - v1.y) / (v2.x - v1.x);
      c = v1.y - (m * v1.x);
      if (i1 & 1) {
        s = m * l + c;
        if (s >= t && s <= b)
          return true;
      }
      if (i1 & 2) {
        s = (t - c) / m;
        if (s >= l && s <= r)
          return true;
      }
      if (i1 & 4) {
        s = m * r + c;
        if (s >= t && s <= b)
          return true;
      }
      if (i1 & 8) {
        s = (b - c) / m;
        if (s >= l && s <= r)
          return true;
      }
    } else {
      if (l == v1.x || r == v1.x)
        return true;
      if (v1.x > l && v1.x < r)
        return true;
    }
  }

  int i2 = b0 ^ b2;
  if (i2 != 0) {
    if (v2.x != v0.x) {
      m = (v2.y - v0.y) / (v2.x - v0.x);
      c = v0.y - (m * v0.x);
      if (i2 & 1) {
        s = m * l + c;
        if (s >= t && s <= b)
          return true;
      }
      if (i2 & 2) {
        s = (t - c) / m;
        if (s >= l && s <= r)
          return true;
      }
      if (i2 & 4) {
        s = m * r + c;
        if (s >= t && s <= b)
          return true;
      }
      if (i2 & 8) {
        s = (b - c) / m;
        if (s >= l && s <= r)
          return true;
      }
    } else {
      if (l == v0.x || r == v0.x)
        return true;
      if (v0.x > l && v0.x < r)
        return true;
    }
  }

  // Bounding box check
  float tbb_l = std::min(v0.x, std::min(v1.x, v2.x));
  float tbb_t = std::min(v0.y, std::min(v1.y, v2.y));
  float tbb_r = std::max(v0.x, std::max(v1.x, v2.x));
  float tbb_b = std::max(v0.y, std::max(v1.y, v2.y));

  if (tbb_l <= l && tbb_r >= r && tbb_t <= t && tbb_b >= b) {
    float v0x = v2.x - v0.x;
    float v0y = v2.y - v0.y;
    float v1x = v1.x - v0.x;
    float v1y = v1.y - v0.y;
    float v2x, v2y;

    float dot00, dot01, dot02, dot11, dot12, invDenom, u, v;

    // Top-left corner
    v2x = l - v0.x;
    v2y = t - v0.y;

    dot00 = v0x * v0x + v0y * v0y;
    dot01 = v0x * v1x + v0y * v1y;
    dot02 = v0x * v2x + v0y * v2y;
    dot11 = v1x * v1x + v1y * v1y;
    dot12 = v1x * v2x + v1y * v2y;

    invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    if (u >= 0 && v >= 0 && (u + v) <= 1)
      return true;

    // Bottom-left corner
    v2x = l - v0.x;
    v2y = b - v0.y;

    dot02 = v0x * v2x + v0y * v2y;
    dot12 = v1x * v2x + v1y * v2y;

    u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    if (u >= 0 && v >= 0 && (u + v) <= 1)
      return true;

    // Bottom-right corner
    v2x = r - v0.x;
    v2y = b - v0.y;

    dot02 = v0x * v2x + v0y * v2y;
    dot12 = v1x * v2x + v1y * v2y;

    u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    if (u >= 0 && v >= 0 && (u + v) <= 1)
      return true;

    // Top-right corner
    v2x = r - v0.x;
    v2y = t - v0.y;

    dot02 = v0x * v2x + v0y * v2y;
    dot12 = v1x * v2x + v1y * v2y;

    u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    if (u >= 0 && v >= 0 && (u + v) <= 1)
      return true;
  }

  return false;
}

void tri_winding(uv_float2 &a, uv_float2 &b, uv_float2 &c) {
  float det = (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

  // If the determinant is negative, the triangle is oriented clockwise
  if (det < 0) {
    // Swap vertices b and c to ensure counter-clockwise winding
    std::swap(b, c);
  }
}

struct Triangle {
  uv_float3 a, b, c;

  Triangle(const uv_float2 &p1, const uv_float2 &q1, const uv_float2 &r1)
      : a({p1.x, p1.y, 0}), b({q1.x, q1.y, 0}), c({r1.x, r1.y, 0}) {}

  Triangle(const uv_float3 &p1, const uv_float3 &q1, const uv_float3 &r1)
      : a(p1), b(q1), c(r1) {}

  void getNormal(uv_float3 &normal) const {
    uv_float3 u = b - a;
    uv_float3 v = c - a;
    normal = normalize(cross(u, v));
  }
};

bool isTriDegenerated(const Triangle &tri) {
  uv_float3 u = tri.a - tri.b;
  uv_float3 v = tri.a - tri.c;
  uv_float3 cr = cross(u, v);
  return fabs(cr.x) < EPSILON && fabs(cr.y) < EPSILON && fabs(cr.z) < EPSILON;
}

int orient3D(const uv_float3 &a, const uv_float3 &b, const uv_float3 &c,
             const uv_float3 &d) {
  Matrix4 _matrix4;
  _matrix4.set(a.x, a.y, a.z, 1, b.x, b.y, b.z, 1, c.x, c.y, c.z, 1, d.x, d.y,
               d.z, 1);
  float det = _matrix4.determinant();

  if (det < -EPSILON)
    return -1;
  else if (det > EPSILON)
    return 1;
  else
    return 0;
}

int orient2D(const uv_float2 &a, const uv_float2 &b, const uv_float2 &c) {
  float det = (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

  if (det < -EPSILON)
    return -1;
  else if (det > EPSILON)
    return 1;
  else
    return 0;
}

int orient2D(const uv_float3 &a, const uv_float3 &b, const uv_float3 &c) {
  uv_float2 a_2d = {a.x, a.y};
  uv_float2 b_2d = {b.x, b.y};
  uv_float2 c_2d = {c.x, c.y};
  return orient2D(a_2d, b_2d, c_2d);
}

void permuteTriLeft(Triangle &tri) {
  uv_float3 tmp = tri.a;
  tri.a = tri.b;
  tri.b = tri.c;
  tri.c = tmp;
}

void permuteTriRight(Triangle &tri) {
  uv_float3 tmp = tri.c;
  tri.c = tri.b;
  tri.b = tri.a;
  tri.a = tmp;
}

void makeTriCounterClockwise(Triangle &tri) {
  if (orient2D(tri.a, tri.b, tri.c) < 0) {
    uv_float3 tmp = tri.c;
    tri.c = tri.b;
    tri.b = tmp;
  }
}

void intersectPlane(const uv_float3 &a, const uv_float3 &b, const uv_float3 &p,
                    const uv_float3 &n, uv_float3 &target) {
  uv_float3 u = b - a;
  uv_float3 v = a - p;
  float dot1 = dot(n, u);
  float dot2 = dot(n, v);
  u = u * (-dot2 / dot1);
  target = a + u;
}

void computeLineIntersection(const Triangle &t1, const Triangle &t2,
                             std::vector<uv_float3> &target) {
  uv_float3 n1, n2;
  t1.getNormal(n1);
  t2.getNormal(n2);

  int o1 = orient3D(t1.a, t1.c, t2.b, t2.a);
  int o2 = orient3D(t1.a, t1.b, t2.c, t2.a);

  uv_float3 i1, i2;

  if (o1 > 0) {
    if (o2 > 0) {
      intersectPlane(t1.a, t1.c, t2.a, n2, i1);
      intersectPlane(t2.a, t2.c, t1.a, n1, i2);
    } else {
      intersectPlane(t1.a, t1.c, t2.a, n2, i1);
      intersectPlane(t1.a, t1.b, t2.a, n2, i2);
    }
  } else {
    if (o2 > 0) {
      intersectPlane(t2.a, t2.b, t1.a, n1, i1);
      intersectPlane(t2.a, t2.c, t1.a, n1, i2);
    } else {
      intersectPlane(t2.a, t2.b, t1.a, n1, i1);
      intersectPlane(t1.a, t1.b, t2.a, n2, i2);
    }
  }

  target.push_back(i1);
  if (distance_to(i1, i2) >= EPSILON) {
    target.push_back(i2);
  }
}

void makeTriAVertexAlone(Triangle &tri, int oa, int ob, int oc) {
  // Permute a, b, c so that a is alone on its side
  if (oa == ob) {
    // c is alone, permute right so c becomes a
    permuteTriRight(tri);
  } else if (oa == oc) {
    // b is alone, permute so b becomes a
    permuteTriLeft(tri);
  } else if (ob != oc) {
    // In case a, b, c have different orientation, put a on positive side
    if (ob > 0) {
      permuteTriLeft(tri);
    } else if (oc > 0) {
      permuteTriRight(tri);
    }
  }
}

void makeTriAVertexPositive(Triangle &tri, const Triangle &other) {
  int o = orient3D(other.a, other.b, other.c, tri.a);
  if (o < 0) {
    std::swap(tri.b, tri.c);
  }
}

bool crossIntersect(Triangle &t1, Triangle &t2, int o1a, int o1b, int o1c,
                    std::vector<uv_float3> *target = nullptr) {
  int o2a = orient3D(t1.a, t1.b, t1.c, t2.a);
  int o2b = orient3D(t1.a, t1.b, t1.c, t2.b);
  int o2c = orient3D(t1.a, t1.b, t1.c, t2.c);

  if (o2a == o2b && o2a == o2c) {
    return false;
  }

  // Make a vertex alone on its side for both triangles
  makeTriAVertexAlone(t1, o1a, o1b, o1c);
  makeTriAVertexAlone(t2, o2a, o2b, o2c);

  // Ensure the vertex on the positive side
  makeTriAVertexPositive(t2, t1);
  makeTriAVertexPositive(t1, t2);

  int o1 = orient3D(t1.a, t1.b, t2.a, t2.b);
  int o2 = orient3D(t1.a, t1.c, t2.c, t2.a);

  if (o1 <= 0 && o2 <= 0) {
    if (target) {
      computeLineIntersection(t1, t2, *target);
    }
    return true;
  }

  return false;
}

void linesIntersect2d(const uv_float3 &a1, const uv_float3 &b1,
                      const uv_float3 &a2, const uv_float3 &b2,
                      uv_float3 &target) {
  float dx1 = a1.x - b1.x;
  float dx2 = a2.x - b2.x;
  float dy1 = a1.y - b1.y;
  float dy2 = a2.y - b2.y;

  float D = dx1 * dy2 - dx2 * dy1;

  float n1 = a1.x * b1.y - a1.y * b1.x;
  float n2 = a2.x * b2.y - a2.y * b2.x;

  target.x = (n1 * dx2 - n2 * dx1) / D;
  target.y = (n1 * dy2 - n2 * dy1) / D;
  target.z = 0;
}

void clipTriangle(const Triangle &t1, const Triangle &t2,
                  std::vector<uv_float3> &target) {
  std::vector<uv_float3> clip = {t1.a, t1.b, t1.c};
  std::vector<uv_float3> output = {t2.a, t2.b, t2.c};
  std::vector<int> orients(output.size() * 3, 0);
  uv_float3 inter;

  for (int i = 0; i < 3; ++i) {
    const int i_prev = (i + 2) % 3;
    std::vector<uv_float3> input;
    std::copy(output.begin(), output.end(), std::back_inserter(input));
    output.clear();

    for (size_t j = 0; j < input.size(); ++j) {
      orients[j] = orient2D(clip[i_prev], clip[i], input[j]);
    }

    for (size_t j = 0; j < input.size(); ++j) {
      const int j_prev = (j - 1 + input.size()) % input.size();

      if (orients[j] >= 0) {
        if (orients[j_prev] < 0) {
          linesIntersect2d(clip[i_prev], clip[i], input[j_prev], input[j],
                           inter);
          output.push_back({inter.x, inter.y, inter.z});
        }
        output.push_back({input[j].x, input[j].y, input[j].z});
      } else if (orients[j_prev] >= 0) {
        linesIntersect2d(clip[i_prev], clip[i], input[j_prev], input[j], inter);
        output.push_back({inter.x, inter.y, inter.z});
      }
    }
  }

  // Clear duplicated points
  for (const auto &point : output) {
    int j = 0;
    bool sameFound = false;
    while (!sameFound && j < target.size()) {
      sameFound = distance_to(point, target[j]) <= 1e-6;
      j++;
    }

    if (!sameFound) {
      target.push_back(point);
    }
  }
}

bool intersectionTypeR1(const Triangle &t1, const Triangle &t2) {
  const uv_float3 &p1 = t1.a;
  const uv_float3 &q1 = t1.b;
  const uv_float3 &r1 = t1.c;
  const uv_float3 &p2 = t2.a;
  const uv_float3 &r2 = t2.c;

  if (orient2D(r2, p2, q1) >= 0) {     // I
    if (orient2D(r2, p1, q1) >= 0) {   // II.a
      if (orient2D(p1, p2, q1) >= 0) { // III.a
        return true;
      } else {
        if (orient2D(p1, p2, r1) >= 0) {   // IV.a
          if (orient2D(q1, r1, p2) >= 0) { // V
            return true;
          }
        }
      }
    }
  } else {
    if (orient2D(r2, p2, r1) >= 0) {     // II.b
      if (orient2D(q1, r1, r2) >= 0) {   // III.b
        if (orient2D(p1, p2, r1) >= 0) { // IV.b (diverges from paper)
          return true;
        }
      }
    }
  }

  return false;
}

bool intersectionTypeR2(const Triangle &t1, const Triangle &t2) {
  const uv_float3 &p1 = t1.a;
  const uv_float3 &q1 = t1.b;
  const uv_float3 &r1 = t1.c;
  const uv_float3 &p2 = t2.a;
  const uv_float3 &q2 = t2.b;
  const uv_float3 &r2 = t2.c;

  if (orient2D(r2, p2, q1) >= 0) {       // I
    if (orient2D(q2, r2, q1) >= 0) {     // II.a
      if (orient2D(p1, p2, q1) >= 0) {   // III.a
        if (orient2D(p1, q2, q1) <= 0) { // IV.a
          return true;
        }
      } else {
        if (orient2D(p1, p2, r1) >= 0) {   // IV.b
          if (orient2D(r2, p2, r1) <= 0) { // V.a
            return true;
          }
        }
      }
    } else {
      if (orient2D(p1, q2, q1) <= 0) {     // III.b
        if (orient2D(q2, r2, r1) >= 0) {   // IV.c
          if (orient2D(q1, r1, q2) >= 0) { // V.b
            return true;
          }
        }
      }
    }
  } else {
    if (orient2D(r2, p2, r1) >= 0) {     // II.b
      if (orient2D(q1, r1, r2) >= 0) {   // III.c
        if (orient2D(r1, p1, p2) >= 0) { // IV.d
          return true;
        }
      } else {
        if (orient2D(q1, r1, q2) >= 0) {   // IV.e
          if (orient2D(q2, r2, r1) >= 0) { // V.c
            return true;
          }
        }
      }
    }
  }

  return false;
}

bool coplanarIntersect(Triangle &t1, Triangle &t2,
                       std::vector<uv_float3> *target = nullptr) {
  uv_float3 normal, u, v;
  t1.getNormal(normal);
  normal = normalize(normal);
  u = normalize(t1.a - t1.b);
  v = cross(normal, u);

  // Move basis to t1.a
  u = u + t1.a;
  v = v + t1.a;
  normal = normal + t1.a;

  Matrix4 _matrix;
  _matrix.set(t1.a.x, u.x, v.x, normal.x, t1.a.y, u.y, v.y, normal.y, t1.a.z,
              u.z, v.z, normal.z, 1, 1, 1, 1);

  Matrix4 _affineMatrix;
  _affineMatrix.set(0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1);

  _matrix.invert(); // Invert the _matrix
  _matrix = _affineMatrix * _matrix;

  // Apply transformation
  apply_matrix4(t1.a, _matrix);
  apply_matrix4(t1.b, _matrix);
  apply_matrix4(t1.c, _matrix);
  apply_matrix4(t2.a, _matrix);
  apply_matrix4(t2.b, _matrix);
  apply_matrix4(t2.c, _matrix);

  makeTriCounterClockwise(t1);
  makeTriCounterClockwise(t2);

  const uv_float3 &p1 = t1.a;
  const uv_float3 &p2 = t2.a;
  const uv_float3 &q2 = t2.b;
  const uv_float3 &r2 = t2.c;

  int o_p2q2 = orient2D(p2, q2, p1);
  int o_q2r2 = orient2D(q2, r2, p1);
  int o_r2p2 = orient2D(r2, p2, p1);

  bool intersecting = false;
  if (o_p2q2 >= 0) {
    if (o_q2r2 >= 0) {
      if (o_r2p2 >= 0) {
        // + + +
        intersecting = true;
      } else {
        // + + -
        intersecting = intersectionTypeR1(t1, t2);
      }
    } else {
      if (o_r2p2 >= 0) {
        // + - +
        permuteTriRight(t2);
        intersecting = intersectionTypeR1(t1, t2);
      } else {
        // + - -
        intersecting = intersectionTypeR2(t1, t2);
      }
    }
  } else {
    if (o_q2r2 >= 0) {
      if (o_r2p2 >= 0) {
        // - + +
        permuteTriLeft(t2);
        intersecting = intersectionTypeR1(t1, t2);
      } else {
        // - + -
        permuteTriLeft(t2);
        intersecting = intersectionTypeR2(t1, t2);
      }
    } else {
      if (o_r2p2 >= 0) {
        // - - +
        permuteTriRight(t2);
        intersecting = intersectionTypeR2(t1, t2);
      } else {
        // - - -
        std::cerr << "Triangles should not be flat." << std::endl;
        return false;
      }
    }
  }

  if (intersecting && target) {
    clipTriangle(t1, t2, *target);

    _matrix.invert();
    // Apply the transform to each target point
    for (int i = 0; i < target->size(); ++i) {
      apply_matrix4(target->at(i), _matrix);
    }
  }

  return intersecting;
}

// Helper function to calculate the area of a polygon
float polygon_area(const std::vector<uv_float3> &polygon) {
  if (polygon.size() < 3)
    return 0.0f; // Not a polygon

  uv_float3 normal = {0.0f, 0.0f, 0.0f}; // Initialize normal vector

  // Calculate the cross product of edges around the polygon
  for (size_t i = 0; i < polygon.size(); ++i) {
    uv_float3 p1 = polygon[i];
    uv_float3 p2 = polygon[(i + 1) % polygon.size()];

    normal = normal + cross(p1, p2); // Accumulate the normal vector
  }

  float area =
      magnitude(normal) / 2.0f; // Area is half the magnitude of the normal
  return area;
}

bool triangle_triangle_intersection(uv_float2 p1, uv_float2 q1, uv_float2 r1,
                                    uv_float2 p2, uv_float2 q2, uv_float2 r2) {
  Triangle t1(p1, q1, r1);
  Triangle t2(p2, q2, r2);

  if (isTriDegenerated(t1) || isTriDegenerated(t2)) {
    // std::cerr << "Degenerated triangles provided, skipping." << std::endl;
    return false;
  }

  int o1a = orient3D(t2.a, t2.b, t2.c, t1.a);
  int o1b = orient3D(t2.a, t2.b, t2.c, t1.b);
  int o1c = orient3D(t2.a, t2.b, t2.c, t1.c);

  std::vector<uv_float3> intersections;
  bool intersects;

  if (o1a == o1b && o1a == o1c) // [[likely]]
  {
    intersects = o1a == 0 && coplanarIntersect(t1, t2, &intersections);
  } else // [[unlikely]]
  {
    intersects = crossIntersect(t1, t2, o1a, o1b, o1c, &intersections);
  }

  if (intersects) {
    float area = polygon_area(intersections);

    // std::cout << "Intersection area: " << area << std::endl;
    if (area < 1e-10f || std::isfinite(area) == false) {
      // std::cout<<"Invalid area: " << area << std::endl;
      return false; // Ignore intersection if the area is too small
    }
  }

  return intersects;
}
