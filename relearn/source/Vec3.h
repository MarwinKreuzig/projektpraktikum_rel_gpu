#pragma once

template<typename T>
struct Vec3 {
	T x;
	T y;
	T z;

	Vec3() : x(0), y(0), z(0) {

	}

	Vec3(T val) : x(val), y(val), z(val) {

	}

	Vec3(T _x, T _y, T _z) :
		x(_x), y(_y), z(_z) {

	}

	bool operator==(const Vec3<T>& other) {
		return (x == other.x) && (y == other.y) && (z == other.z);
	}

	template <typename K>
	T& operator[](const K& index) {
		if (index == 0) {
			return x;
		}
		if (index == 1) {
			return y;
		}
		assert(index == 2);
		return z;
	}

	struct less {
		bool operator() (const Vec3<T>& lhs, const Vec3<T>& rhs) const {
			return  lhs.x < rhs.x ||
				(lhs.x == rhs.x && lhs.y < rhs.y) ||
				(lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z);
		}
	};
};

using Vec3d = Vec3<double>;
