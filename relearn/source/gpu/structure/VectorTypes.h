#pragma once

namespace gpu {

    template <typename T>
    struct Vec3 {
        T x;
        T y;
        T z;

        Vec3(T x, T y, T z)
            : x(x), y(y), z(z)
        {}

        Vec3() {}
    };

    using Vec3d = Vec3<double>;
};