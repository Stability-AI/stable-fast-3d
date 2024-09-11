#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

const float EPSILON = 1e-7f;

// Structure to represent a 2D point or vector
union alignas(8) uv_float2 {
  struct {
    float x, y;
  };

  float data[2];

  float &operator[](size_t idx) {
    if (idx > 1)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  const float &operator[](size_t idx) const {
    if (idx > 1)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  bool operator==(const uv_float2 &rhs) const {
    return x == rhs.x && y == rhs.y;
  }
};

// Do not align as this is specifically tweaked for BVHNode
union uv_float3 {
  struct {
    float x, y, z;
  };

  float data[3];

  float &operator[](size_t idx) {
    if (idx > 3)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  const float &operator[](size_t idx) const {
    if (idx > 3)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  bool operator==(const uv_float3 &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
};

union alignas(16) uv_float4 {
  struct {
    float x, y, z, w;
  };

  float data[4];

  float &operator[](size_t idx) {
    if (idx > 3)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  const float &operator[](size_t idx) const {
    if (idx > 3)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  bool operator==(const uv_float4 &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w;
  }
};

union alignas(8) uv_int2 {
  struct {
    int x, y;
  };

  int data[2];

  int &operator[](size_t idx) {
    if (idx > 1)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  const int &operator[](size_t idx) const {
    if (idx > 1)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  bool operator==(const uv_int2 &rhs) const { return x == rhs.x && y == rhs.y; }
};

union alignas(4) uv_int3 {
  struct {
    int x, y, z;
  };

  int data[3];

  int &operator[](size_t idx) {
    if (idx > 2)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  const int &operator[](size_t idx) const {
    if (idx > 2)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  bool operator==(const uv_int3 &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
};

union alignas(16) uv_int4 {
  struct {
    int x, y, z, w;
  };

  int data[4];

  int &operator[](size_t idx) {
    if (idx > 3)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  const int &operator[](size_t idx) const {
    if (idx > 3)
      throw std::runtime_error("bad index");
    return data[idx];
  }

  bool operator==(const uv_int4 &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w;
  }
};

inline float calc_mean(float a, float b, float c) { return (a + b + c) / 3; }

// Create a triangle centroid
inline uv_float2 triangle_centroid(const uv_float2 &v0, const uv_float2 &v1,
                                   const uv_float2 &v2) {
  return {calc_mean(v0.x, v1.x, v2.x), calc_mean(v0.y, v1.y, v2.y)};
}

inline uv_float3 triangle_centroid(const uv_float3 &v0, const uv_float3 &v1,
                                   const uv_float3 &v2) {
  return {calc_mean(v0.x, v1.x, v2.x), calc_mean(v0.y, v1.y, v2.y),
          calc_mean(v0.z, v1.z, v2.z)};
}

// Helper functions for vector math
inline uv_float2 operator-(const uv_float2 &a, const uv_float2 &b) {
  return {a.x - b.x, a.y - b.y};
}

inline uv_float3 operator-(const uv_float3 &a, const uv_float3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline uv_float2 operator+(const uv_float2 &a, const uv_float2 &b) {
  return {a.x + b.x, a.y + b.y};
}

inline uv_float3 operator+(const uv_float3 &a, const uv_float3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline uv_float2 operator*(const uv_float2 &a, float scalar) {
  return {a.x * scalar, a.y * scalar};
}

inline uv_float3 operator*(const uv_float3 &a, float scalar) {
  return {a.x * scalar, a.y * scalar, a.z * scalar};
}

inline float dot(const uv_float2 &a, const uv_float2 &b) {
  return a.x * b.x + a.y * b.y;
}

inline float dot(const uv_float3 &a, const uv_float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float cross(const uv_float2 &a, const uv_float2 &b) {
  return a.x * b.y - a.y * b.x;
}

inline uv_float3 cross(const uv_float3 &a, const uv_float3 &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline uv_float2 abs_vec(const uv_float2 &v) {
  return {std::abs(v.x), std::abs(v.y)};
}

inline uv_float2 min_vec(const uv_float2 &a, const uv_float2 &b) {
  return {std::min(a.x, b.x), std::min(a.y, b.y)};
}

inline uv_float2 max_vec(const uv_float2 &a, const uv_float2 &b) {
  return {std::max(a.x, b.x), std::max(a.y, b.y)};
}

inline float distance_to(const uv_float2 &a, const uv_float2 &b) {
  return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

inline float distance_to(const uv_float3 &a, const uv_float3 &b) {
  return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) +
                   std::pow(a.z - b.z, 2));
}

inline uv_float2 normalize(const uv_float2 &v) {
  float len = std::sqrt(v.x * v.x + v.y * v.y);
  return {v.x / len, v.y / len};
}

inline uv_float3 normalize(const uv_float3 &v) {
  float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  return {v.x / len, v.y / len, v.z / len};
}

inline float magnitude(const uv_float3 &v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

struct Matrix4 {
  std::array<std::array<float, 4>, 4> m;

  Matrix4() {
    for (auto &row : m) {
      row.fill(0.0f);
    }
    m[3][3] = 1.0f; // Identity matrix for 4th row and column
  }

  void set(float m00, float m01, float m02, float m03, float m10, float m11,
           float m12, float m13, float m20, float m21, float m22, float m23,
           float m30, float m31, float m32, float m33) {
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[0][3] = m03;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[1][3] = m13;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
    m[2][3] = m23;
    m[3][0] = m30;
    m[3][1] = m31;
    m[3][2] = m32;
    m[3][3] = m33;
  }

  float determinant() const {
    return m[0][3] * m[1][2] * m[2][1] * m[3][0] -
           m[0][2] * m[1][3] * m[2][1] * m[3][0] -
           m[0][3] * m[1][1] * m[2][2] * m[3][0] +
           m[0][1] * m[1][3] * m[2][2] * m[3][0] +
           m[0][2] * m[1][1] * m[2][3] * m[3][0] -
           m[0][1] * m[1][2] * m[2][3] * m[3][0] -
           m[0][3] * m[1][2] * m[2][0] * m[3][1] +
           m[0][2] * m[1][3] * m[2][0] * m[3][1] +
           m[0][3] * m[1][0] * m[2][2] * m[3][1] -
           m[0][0] * m[1][3] * m[2][2] * m[3][1] -
           m[0][2] * m[1][0] * m[2][3] * m[3][1] +
           m[0][0] * m[1][2] * m[2][3] * m[3][1] +
           m[0][3] * m[1][1] * m[2][0] * m[3][2] -
           m[0][1] * m[1][3] * m[2][0] * m[3][2] -
           m[0][3] * m[1][0] * m[2][1] * m[3][2] +
           m[0][0] * m[1][3] * m[2][1] * m[3][2] +
           m[0][1] * m[1][0] * m[2][3] * m[3][2] -
           m[0][0] * m[1][1] * m[2][3] * m[3][2] -
           m[0][2] * m[1][1] * m[2][0] * m[3][3] +
           m[0][1] * m[1][2] * m[2][0] * m[3][3] +
           m[0][2] * m[1][0] * m[2][1] * m[3][3] -
           m[0][0] * m[1][2] * m[2][1] * m[3][3] -
           m[0][1] * m[1][0] * m[2][2] * m[3][3] +
           m[0][0] * m[1][1] * m[2][2] * m[3][3];
  }

  Matrix4 operator*(const Matrix4 &other) const {
    Matrix4 result;
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        result.m[row][col] =
            m[row][0] * other.m[0][col] + m[row][1] * other.m[1][col] +
            m[row][2] * other.m[2][col] + m[row][3] * other.m[3][col];
      }
    }
    return result;
  }

  Matrix4 operator*(float scalar) const {
    Matrix4 result = *this;
    for (auto &row : result.m) {
      for (auto &element : row) {
        element *= scalar;
      }
    }
    return result;
  }

  Matrix4 operator+(const Matrix4 &other) const {
    Matrix4 result;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        result.m[i][j] = m[i][j] + other.m[i][j];
      }
    }
    return result;
  }

  Matrix4 operator-(const Matrix4 &other) const {
    Matrix4 result;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        result.m[i][j] = m[i][j] - other.m[i][j];
      }
    }
    return result;
  }

  float trace() const { return m[0][0] + m[1][1] + m[2][2] + m[3][3]; }

  Matrix4 identity() const {
    Matrix4 identity;
    identity.set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    return identity;
  }

  Matrix4 power(int exp) const {
    if (exp == 0)
      return identity();
    if (exp == 1)
      return *this;

    Matrix4 result = *this;
    for (int i = 1; i < exp; ++i) {
      result = result * (*this);
    }
    return result;
  }

  void print() {
    // Print all entries in 4 rows with 4 columns
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        std::cout << m[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  bool invert() {
    double inv[16], det;
    double mArr[16];

    // Convert the matrix to a 1D array for easier manipulation
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        mArr[i * 4 + j] = static_cast<double>(m[i][j]);
      }
    }

    inv[0] = mArr[5] * mArr[10] * mArr[15] - mArr[5] * mArr[11] * mArr[14] -
             mArr[9] * mArr[6] * mArr[15] + mArr[9] * mArr[7] * mArr[14] +
             mArr[13] * mArr[6] * mArr[11] - mArr[13] * mArr[7] * mArr[10];

    inv[4] = -mArr[4] * mArr[10] * mArr[15] + mArr[4] * mArr[11] * mArr[14] +
             mArr[8] * mArr[6] * mArr[15] - mArr[8] * mArr[7] * mArr[14] -
             mArr[12] * mArr[6] * mArr[11] + mArr[12] * mArr[7] * mArr[10];

    inv[8] = mArr[4] * mArr[9] * mArr[15] - mArr[4] * mArr[11] * mArr[13] -
             mArr[8] * mArr[5] * mArr[15] + mArr[8] * mArr[7] * mArr[13] +
             mArr[12] * mArr[5] * mArr[11] - mArr[12] * mArr[7] * mArr[9];

    inv[12] = -mArr[4] * mArr[9] * mArr[14] + mArr[4] * mArr[10] * mArr[13] +
              mArr[8] * mArr[5] * mArr[14] - mArr[8] * mArr[6] * mArr[13] -
              mArr[12] * mArr[5] * mArr[10] + mArr[12] * mArr[6] * mArr[9];

    inv[1] = -mArr[1] * mArr[10] * mArr[15] + mArr[1] * mArr[11] * mArr[14] +
             mArr[9] * mArr[2] * mArr[15] - mArr[9] * mArr[3] * mArr[14] -
             mArr[13] * mArr[2] * mArr[11] + mArr[13] * mArr[3] * mArr[10];

    inv[5] = mArr[0] * mArr[10] * mArr[15] - mArr[0] * mArr[11] * mArr[14] -
             mArr[8] * mArr[2] * mArr[15] + mArr[8] * mArr[3] * mArr[14] +
             mArr[12] * mArr[2] * mArr[11] - mArr[12] * mArr[3] * mArr[10];

    inv[9] = -mArr[0] * mArr[9] * mArr[15] + mArr[0] * mArr[11] * mArr[13] +
             mArr[8] * mArr[1] * mArr[15] - mArr[8] * mArr[3] * mArr[13] -
             mArr[12] * mArr[1] * mArr[11] + mArr[12] * mArr[3] * mArr[9];

    inv[13] = mArr[0] * mArr[9] * mArr[14] - mArr[0] * mArr[10] * mArr[13] -
              mArr[8] * mArr[1] * mArr[14] + mArr[8] * mArr[2] * mArr[13] +
              mArr[12] * mArr[1] * mArr[10] - mArr[12] * mArr[2] * mArr[9];

    inv[2] = mArr[1] * mArr[6] * mArr[15] - mArr[1] * mArr[7] * mArr[14] -
             mArr[5] * mArr[2] * mArr[15] + mArr[5] * mArr[3] * mArr[14] +
             mArr[13] * mArr[2] * mArr[7] - mArr[13] * mArr[3] * mArr[6];

    inv[6] = -mArr[0] * mArr[6] * mArr[15] + mArr[0] * mArr[7] * mArr[14] +
             mArr[4] * mArr[2] * mArr[15] - mArr[4] * mArr[3] * mArr[14] -
             mArr[12] * mArr[2] * mArr[7] + mArr[12] * mArr[3] * mArr[6];

    inv[10] = mArr[0] * mArr[5] * mArr[15] - mArr[0] * mArr[7] * mArr[13] -
              mArr[4] * mArr[1] * mArr[15] + mArr[4] * mArr[3] * mArr[13] +
              mArr[12] * mArr[1] * mArr[7] - mArr[12] * mArr[3] * mArr[5];

    inv[14] = -mArr[0] * mArr[5] * mArr[14] + mArr[0] * mArr[6] * mArr[13] +
              mArr[4] * mArr[1] * mArr[14] - mArr[4] * mArr[2] * mArr[13] -
              mArr[12] * mArr[1] * mArr[6] + mArr[12] * mArr[2] * mArr[5];

    inv[3] = -mArr[1] * mArr[6] * mArr[11] + mArr[1] * mArr[7] * mArr[10] +
             mArr[5] * mArr[2] * mArr[11] - mArr[5] * mArr[3] * mArr[10] -
             mArr[9] * mArr[2] * mArr[7] + mArr[9] * mArr[3] * mArr[6];

    inv[7] = mArr[0] * mArr[6] * mArr[11] - mArr[0] * mArr[7] * mArr[10] -
             mArr[4] * mArr[2] * mArr[11] + mArr[4] * mArr[3] * mArr[10] +
             mArr[8] * mArr[2] * mArr[7] - mArr[8] * mArr[3] * mArr[6];

    inv[11] = -mArr[0] * mArr[5] * mArr[11] + mArr[0] * mArr[7] * mArr[9] +
              mArr[4] * mArr[1] * mArr[11] - mArr[4] * mArr[3] * mArr[9] -
              mArr[8] * mArr[1] * mArr[7] + mArr[8] * mArr[3] * mArr[5];

    inv[15] = mArr[0] * mArr[5] * mArr[10] - mArr[0] * mArr[6] * mArr[9] -
              mArr[4] * mArr[1] * mArr[10] + mArr[4] * mArr[2] * mArr[9] +
              mArr[8] * mArr[1] * mArr[6] - mArr[8] * mArr[2] * mArr[5];

    det = mArr[0] * inv[0] + mArr[1] * inv[4] + mArr[2] * inv[8] +
          mArr[3] * inv[12];

    if (fabs(det) < 1e-6) {
      return false;
    }

    det = 1.0 / det;

    for (int i = 0; i < 16; i++) {
      inv[i] *= det;
    }

    // Convert the 1D array back to the 4x4 matrix
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        m[i][j] = static_cast<float>(inv[i * 4 + j]);
      }
    }

    return true;
  }
};

inline void apply_matrix4(uv_float3 &v, const Matrix4 matrix) {
  float newX = v.x * matrix.m[0][0] + v.y * matrix.m[0][1] +
               v.z * matrix.m[0][2] + matrix.m[0][3];
  float newY = v.x * matrix.m[1][0] + v.y * matrix.m[1][1] +
               v.z * matrix.m[1][2] + matrix.m[1][3];
  float newZ = v.x * matrix.m[2][0] + v.y * matrix.m[2][1] +
               v.z * matrix.m[2][2] + matrix.m[2][3];
  float w = v.x * matrix.m[3][0] + v.y * matrix.m[3][1] + v.z * matrix.m[3][2] +
            matrix.m[3][3];

  if (std::fabs(w) > EPSILON) {
    newX /= w;
    newY /= w;
    newZ /= w;
  }

  v.x = newX;
  v.y = newY;
  v.z = newZ;
}
