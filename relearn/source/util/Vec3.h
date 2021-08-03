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

#include "../Config.h"
#include "../util/RelearnException.h"

#include <spdlog/fmt/bundled/ostream.h>

#include <algorithm>
#include <ostream>
#include <type_traits>

/**
 * @brief A Vec3 holds three different values of type T and allows computations via operators.
 * @tparam T The type that shall be stored inside this class. Is required to fullfill std::is_arithmetic_v<T>
*/
template <typename T>
class Vec3 {
    T x{ 0 };
    T y{ 0 };
    T z{ 0 };

    static_assert(std::is_arithmetic_v<T>);

public:
    /**
     * @brief Constructs a new instance and initializes all values with 0
     */
    Vec3() = default;
    ~Vec3() = default;

    /**
     * @brief Constructs a new instance and initializes all values with val
     * @param val The value that is used to initialize all values 
     */
    explicit Vec3(const T& val) noexcept
        : x(val)
        , y(val)
        , z(val) {
    }

    /**
     * @brief Constructs a new instance and initializes all values
     * @param _x The value for x
     * @param _y The value for y
     * @param _z The value for z
     */
    Vec3(const T& _x, const T& _y, const T& _z) noexcept
        : x(_x)
        , y(_y)
        , z(_z) {
    }

    Vec3(const Vec3<T>& other) = default;
    Vec3<T>& operator=(const Vec3<T>& other) = default;

    Vec3(Vec3<T>&& other) noexcept = default;
    Vec3<T>& operator=(Vec3<T>&& other) noexcept = default;

    /**
     * @brief Returns a constant reference to the x component. The reference is only invalidated by destruction of the object
     * @return The x value
     */
    [[nodiscard]] const T& get_x() const noexcept {
        return x;
    }

    /**
     * @brief Returns a constant reference to the y component. The reference is only invalidated by destruction of the object
     * @return The y value
     */
    [[nodiscard]] const T& get_y() const noexcept {
        return y;
    }

    /**
     * @brief Returns a constant reference to the z component. The reference is only invalidated by destruction of the object
     * @return The z value
     */
    [[nodiscard]] const T& get_z() const noexcept {
        return z;
    }

    /**
     * @brief Sets the x component to the new value
     * @param _x The new value
     */
    void set_x(const T& _x) noexcept {
        x = _x;
    }

    /**
     * @brief Sets the y component to the new value
     * @param _y The new value
     */
    void set_y(const T& _y) noexcept {
        y = _y;
    }

    /**
     * @brief Sets the z component to the new value
     * @param _z The new value
     */
    void set_z(const T& _z) noexcept {
        z = _z;
    }

    /**
     * @brief Casts to current object to an object of type Vec3<K>. Uses static_cast<K> component wise
     * @tparam K The new type of the components
     * @return A castet version of the current object
     */
    template <typename K>
    [[nodiscard]] explicit operator Vec3<K>() const noexcept {
        Vec3<K> res{ static_cast<K>(x), static_cast<K>(y), static_cast<K>(z) };
        return res;
    }

    /**
     * @brief Compares the current object to other and checks for equality
     * @param other The other vector
     * @return True iff all components are equal
     */
    [[nodiscard]] bool operator==(const Vec3<T>& other) const noexcept {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }

    /**
     * @brief Calculates the difference between two vectors and returns it as a newly created object
     * @param lhs The vector from which should be subtracted
     * @param rhs The vector that should be subtracted
     * @return The difference of both vectors as a new object
     */
    [[nodiscard]] friend Vec3<T> operator-(const Vec3<T>& lhs, const Vec3<T>& rhs) noexcept {
        return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
    }

    /**
     * @brief Calculates the sum of two vectors and returns it as a newly created object
     * @param lhs The vector to which should be summed
     * @param rhs The vector that should be summed
     * @return The sum of both vectors as a new object
     */
    [[nodiscard]] friend Vec3<T> operator+(const Vec3<T>& lhs, const Vec3<T>& rhs) noexcept {
        return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
    }

    /**
     * @brief Componentwise adds the scalar value and returns the sum as a newly created object
     * @param scalar The value that should be added to each component
     * @return The sum as a new object
     */
    [[nodiscard]] Vec3<T> operator+(const T& scalar) const noexcept {
        Vec3<T> res = *this;
        res += scalar;
        return res;
    }

    /**
     * @brief Componentwise subtracts the scalar value and returns the difference as a newly created object
     * @param scalar The value that should be subtracted from each component
     * @return The difference as a new object
     */
    [[nodiscard]] Vec3<T> operator-(const T& scalar) const noexcept {
        Vec3<T> res = *this;
        res -= scalar;
        return res;
    }

    /**
     * @brief Componentwise multiplies the scalar value and returns the product as a newly created object
     * @param scalar The value that should be multiplied to each component
     * @return The product as a new object
     */
    [[nodiscard]] Vec3<T> operator*(const T& scalar) const noexcept {
        Vec3<T> res = *this;
        res *= scalar;
        return res;
    }

    /**
     * @brief Componentwise divides by the scalar value and returns the quotient as a newly created object
     * @param scalar The value that should be divided by, is not checked for 0
     * @return The quotient as a new object
     */
    [[nodiscard]] Vec3<T> operator/(const T& scalar) const noexcept {
        Vec3<T> res = *this;
        res /= scalar;
        return res;
    }

    /**
     * @brief Rounds the current object componentwise to a larger multiple of value.
     *      This is effectively an ugly function and should only be used with enough care
     * @param value The value of which the components should be rounded to (a multiple of)
     */
    void round_to_larger_multiple(const T& value) noexcept {
        x = ceil((x - Constants::eps) / value) * value;
        y = ceil((y - Constants::eps) / value) * value;
        z = ceil((z - Constants::eps) / value) * value;
    }

    /**
     * @brief Floors the current vector and returns the results in a newly created object.
     *      Can only be used if the values are non-negative
     * @exception Throws a RelearnException if any of the components are < 0
     * @return A newly created object with the floored values
     */
    [[nodiscard]] Vec3<size_t> floor_componentwise() const /*noexcept*/ {
        RelearnException::check(x >= 0, "floor_componentwise must be used on a positive vector: x");
        RelearnException::check(y >= 0, "floor_componentwise must be used on a positive vector: y");
        RelearnException::check(z >= 0, "floor_componentwise must be used on a positive vector: z");

        const auto floored_x = static_cast<size_t>(floor(x));
        const auto floored_y = static_cast<size_t>(floor(y));
        const auto floored_z = static_cast<size_t>(floor(z));

        return Vec3<size_t>(floored_x, floored_y, floored_z);
    }

    /**
     * @brief Calculates the (signed) volume of the cube with the sidelength of the current object
     * @return The volume, calculated by x * y * z
     */
    [[nodiscard]] T get_volume() const noexcept {
        return x * y * z;
    }

    /**
     * @brief Componentwise multiplies the scalar value and changes the current object
     * @param scalar The value that should be multiplied to each component
     * @return A reference to the current object
     */
    Vec3<T>& operator*=(const T& scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    /**
     * @brief Componentwise divides by the scalar value and changes the current object
     * @param scalar The value that should be divided by for each component
     * @return A reference to the current object
     */
    Vec3<T>& operator/=(const T& scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    /**
     * @brief Componentwise adds the scalar value and changes the current object
     * @param scalar The value that should be added to each component
     * @return A reference to the current object
     */
    Vec3<T>& operator+=(const T& scalar) noexcept {
        x += scalar;
        y += scalar;
        z += scalar;
        return *this;
    }

    /**
     * @brief Componentwise adds the other vector and changes the current object
     * @param other The other vector that should be added componentwise
     * @return A reference to the current object
     */
    Vec3<T>& operator+=(const Vec3<T>& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    /**
     * @brief Componentwise subtracts the scalar value and changes the current object
     * @param scalar The value that should be subtracted from each component
     * @return A reference to the current object
     */
    Vec3<T>& operator-=(const T& scalar) noexcept {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        return *this;
    }

    /**
     * @brief Componentwise subtracts the other vector and changes the current object
     * @param other The other vector that should be subtracted componentwise
     * @return A reference to the current object
     */
    Vec3<T>& operator-=(const Vec3<T>& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    /**
	 * @brief Calculated the p norm of the current obejct
     * @param p The exponent of the norm, must be >= 1.0
     * @exception Throws a RelearnException if p < 1.0
     * @return The calculated norm
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

    /**
     * @brief Calculates the 2-norm of the vector
     * @return The calculated 2-norm
     */
    [[nodiscard]] double calculate_2_norm() const noexcept {
        const auto xx = x * x;
        const auto yy = y * y;
        const auto zz = z * z;

        const auto sum = xx + yy + zz;
        const auto norm = sqrt(sum);
        return norm;
    }

    /**
     * @brief Calculates the squared 2-norm of the vector, i.e., ||this||^2_2
     * @return The squared calculated 2-norm
     */
    [[nodiscard]] double calculate_squared_2_norm() const noexcept {
        const auto xx = x * x;
        const auto yy = y * y;
        const auto zz = z * z;

        const auto sum = xx + yy + zz;
        return sum;
    }

    /**
     * @brief Calculates the maximum of both vectors componentwise and changes the current object
     * @param other The other vector
     */
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

    /**
     * @brief Calculates the minimum of both vectors componentwise and changes the current object
     * @param other The other vector
     */
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

    /**
     * @brief Returns the maximum out of x, y, and z
     * @return The maximum out of x, y, and z
     */
    [[nodiscard]] T get_maximum() const noexcept {
        return std::max({ x, y, z });
    }

    /**
     * @brief Returns the minimum out of x, y, and z
     * @return The minimum out of x, y, and z
     */
    [[nodiscard]] T get_minimum() const noexcept {
        return std::min({ x, y, z });
    }

    /**
     * @brief Provides a linear order on the vectors by looking at x, in case of equality on y, in case of equality on z
     * @param other The other vector that should be compared
     * @return True iff the current vector is smaller than the other
     */
    [[nodiscard]] bool operator<(const Vec3<T>& other) const noexcept {
        return x < other.x || (x == other.x && y < other.y) || (x == other.x && y == other.y && z < other.z);
    }

    /**
     * @brief This struct is here to allow Vec3<T> in std::map, etc.
     */
    struct less {
        /**
         * @brief Calls lhs < rhs
         * @param lhs One Vec3 that should be compared
         * @param rhs The second Vec3 that should be compared
         * @return True iff lhs < rhs
         */
        [[nodiscard]] bool operator()(const Vec3<T>& lhs, const Vec3<T>& rhs) const noexcept {
            return lhs < rhs;
        }
    };

    /**
     * @brief Prints the object to the ostream in the format (x, y, z)
     * @param os The stream to which the object should be printed
     * @param vec The object that should be printed
     * @return A reference to os that allows chaining.
     *      Is not marked as [[nodiscard]] as that typically does happen when chaining <<
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const Vec3<T>& vec) {
        const auto& x = vec.get_x();
        const auto& y = vec.get_y();
        const auto& z = vec.get_z();

        output_stream << '(' << x << ", " << y << ", " << z << ')';

        return output_stream;
    }
};

using Vec3d = Vec3<double>;
using Vec3s = Vec3<size_t>;
