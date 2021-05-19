#pragma once

#include <algorithm>
#include <math.h>

// Position of vertex
struct Position {
    double x, y, z;

    Position()
        : x(0.0)
        , y(0.0)
        , z(0.0) { }

    Position(double x, double y, double z)
        : x(x)
        , y(y)
        , z(z) {
    }

    struct less {
        bool operator()(const Position& lhs, const Position& rhs) const {
            return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y) || (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z);
        }
    };

    void MinForEachCoordinate(const Position& pos) {
        x = std::min(x, pos.x);
        y = std::min(y, pos.y);
        z = std::min(z, pos.z);
    }

    void Add(const Position& pos) {
        x += pos.x;
        y += pos.y;
        z += pos.z;
    }

    double CalcEuclDist(const Position& other) {
        auto diff_x = (x - other.x) * (x - other.x);
        auto diff_y = (y - other.y) * (y - other.y);
        auto diff_z = (z - other.z) * (z - other.z);

        return sqrt(diff_x + diff_y + diff_z);
    }
};
