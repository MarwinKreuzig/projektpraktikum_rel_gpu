#pragma once

namespace gpu {

template <typename T>
struct Vec3 {
    /**
     * This struct is a version of the CPU Vec3 struct that can be included in CUDA source files. Like its CPU counterpart, it's a
     * three dimensional mathematical vector, although this version lacks most of the methods that actually implement the
     * functionality.
     * */

    T x;
    T y;
    T z;

    Vec3(T x, T y, T z)
        : x(x)
        , y(y)
        , z(z) { }

    Vec3() { }

    bool operator!=(const Vec3& rhs) const {
        return (x != rhs.x) || (y != rhs.y) || (z != rhs.z);
    }

    bool operator==(const Vec3& rhs) const {
        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
    }
};

using Vec3d = Vec3<double>;
};