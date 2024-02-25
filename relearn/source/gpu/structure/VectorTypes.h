#pragma once

namespace gpu {

template <typename T>
struct Vec3 {
    T x;
    T y;
    T z;

    Vec3(T x, T y, T z)
        : x(x)
        , y(y)
        , z(z) { }

    Vec3()
        : x(0)
        , y(0)
        , z(0) { }

    bool operator!=(const Vec3& rhs) const {
        return (x != rhs.x) || (y != rhs.y) || (z != rhs.z);
    }

    bool operator==(const Vec3& rhs) const {
        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
    }
};

using Vec3d = Vec3<double>;
};