/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "Config.h"
#include "RelearnException.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

template <typename T>
class Vec3 {
    T x;
    T y;
    T z;

public:
    Vec3() noexcept
        : x(0)
        , y(0)
        , z(0) {
    }

    ~Vec3() = default;

    explicit Vec3(const T& val) noexcept
        : x(val)
        , y(val)
        , z(val) {
    }

    Vec3(const T& _x, const T& _y, const T& _z) noexcept
        : x(_x)
        , y(_y)
        , z(_z) {
    }

    Vec3(const Vec3<T>& other) = default;
    Vec3<T>& operator=(const Vec3<T>& other) = default;

    Vec3(Vec3<T>&& other) noexcept = default;
    Vec3<T>& operator=(Vec3<T>&& other) noexcept = default;

    [[nodiscard]] const T& get_x() const noexcept {
        return x;
    }

    [[nodiscard]] const T& get_y() const noexcept {
        return y;
    }

    [[nodiscard]] const T& get_z() const noexcept {
        return z;
    }

    void set_x(const T& _x) noexcept {
        x = _x;
    }

    void set_y(const T& _y) noexcept {
        y = _y;
    }

    void set_z(const T& _z) noexcept {
        z = _z;
    }

    template <typename K>
    explicit operator Vec3<K>() const noexcept {
        Vec3<K> res{ static_cast<K>(x), static_cast<K>(y), static_cast<K>(z) };
        return res;
    }

    bool operator==(const Vec3<T>& other) const /*noexcept*/ {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }

    template <typename K>
    T& operator[](const K& index) /*noexcept*/ {
        if (index == 0) {
            return x;
        }
        if (index == 1) {
            return y;
        }

        RelearnException::check(index == 2, "indexing with number unequal to 0, 1, 2");
        return z;
    }

    template <typename K>
    const T& operator[](const K& index) const /*noexcept*/ {
        if (index == 0) {
            return x;
        }
        if (index == 1) {
            return y;
        }

        RelearnException::check(index == 2, "indexing with number unequal to 0, 1, 2");
        return z;
    }

    friend Vec3<T> operator-(const Vec3<T>& lhs, const Vec3<T>& rhs) noexcept {
        return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
    }

    friend Vec3<T> operator+(const Vec3<T>& lhs, const Vec3<T>& rhs) noexcept {
        return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
    }

    Vec3<T> operator+(const T& scalar) const noexcept {
        Vec3<T> res = *this;
        res += scalar;
        return res;
    }

    Vec3<T> operator-(const T& scalar) const noexcept {
        Vec3<T> res = *this;
        res -= scalar;
        return res;
    }

    Vec3<T> operator*(const double scalar) const noexcept {
        Vec3<T> res = *this;
        res *= scalar;
        return res;
    }

    Vec3<T> operator/(const double scalar) const noexcept {
        Vec3<T> res = *this;
        res /= scalar;
        return res;
    }

    void round_to_larger_multiple(const T& multiple) noexcept {
        x = ceil((x - Constants::eps) / multiple) * multiple;
        y = ceil((y - Constants::eps) / multiple) * multiple;
        z = ceil((z - Constants::eps) / multiple) * multiple;
    }

    [[nodiscard]] Vec3<size_t> floor_componentwise() const /*noexcept*/ {
        RelearnException::check(x >= 0, "floor_componentwise must be used on a positive vector: x");
        RelearnException::check(y >= 0, "floor_componentwise must be used on a positive vector: y");
        RelearnException::check(z >= 0, "floor_componentwise must be used on a positive vector: z");

        const auto floored_x = static_cast<size_t>(floor(x));
        const auto floored_y = static_cast<size_t>(floor(y));
        const auto floored_z = static_cast<size_t>(floor(z));

        return Vec3<size_t>(floored_x, floored_y, floored_z);
    }

    [[nodiscard]] T get_volume() const noexcept {
        return x * y * z;
    }

    Vec3<T>& operator*=(const T& scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vec3<T>& operator/=(const T& scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    Vec3<T>& operator+=(const T& scalar) noexcept {
        x += scalar;
        y += scalar;
        z += scalar;
        return *this;
    }

    Vec3<T>& operator-=(const T& scalar) noexcept {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        return *this;
    }

    /**
	 * Calculates the p-Norm. Assumens p >= 1.0
	 */
    [[nodiscard]] double calculate_p_norm(const double p) const {
        RelearnException::check(p >= 1.0, "p-norm is only valid for p >= 1.0");

        const auto xx = pow(std::abs(static_cast<double>(x)), p);
        const auto yy = pow(std::abs(static_cast<double>(y)), p);
        const auto zz = pow(std::abs(static_cast<double>(z)), p);

        const auto sum = xx + yy + zz;
        const auto norm = pow(sum, 1.0 / p);
        return norm;
    }

    void calculate_componentwise_maximum(const Vec3<T>& other) noexcept {
        if (other.x > x) {
            x = other.x;
        }
        if (other.y > y) {
            y = other.y;
        }
        if (other.z > z) {
            z = other.z;
        }
    }

    void calculate_componentwise_minimum(const Vec3<T>& other) noexcept {
        if (other.x < x) {
            x = other.x;
        }
        if (other.y < y) {
            y = other.y;
        }
        if (other.z < z) {
            z = other.z;
        }
    }

    [[nodiscard]] T get_maximum() const noexcept {
        return std::max({ x, y, z });
    }

    [[nodiscard]] T get_minimum() const noexcept {
        return std::min({ x, y, z });
    }

    bool operator<(const Vec3<T>& other) const noexcept {
        return x < other.x || (x == other.x && y < other.y) || (x == other.x && y == other.y && z < other.z);
    }

    struct less {
        bool operator()(const Vec3<T>& lhs, const Vec3<T>& rhs) const noexcept {
            return lhs < rhs;
        }
    };
};

using Vec3d = Vec3<double>;
using Vec3s = Vec3<size_t>;
