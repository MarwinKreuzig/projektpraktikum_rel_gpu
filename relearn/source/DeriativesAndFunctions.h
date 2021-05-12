/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <math.h>
#include "Vec3.h"

namespace Deriatives {
//this file contains all x derivatives of e that are required for the Fast Gauss algorithm
inline double deriative1(double t) {
    return -2 * t * exp(-pow(t, 2));
}

inline double deriative2(double t) {
    return ((4 * pow(t, 2)) - 2) * exp(-pow(t, 2));
}

inline double deriative3(double t) {
    return -((8 * pow(t, 3)) - (12 * t)) * exp(-pow(t, 2));
}

inline double deriative4(double t) {
    return ((16 * pow(t, 4)) - (48 * pow(t, 2)) + 12) * exp(-pow(t, 2));
}

inline double deriative5(double t) {
    return -((32 * pow(t, 5)) - (160 * pow(t, 3)) + (120 * t)) * exp(-pow(t, 2));
}

inline double deriative6(double t) {
    return ((64 * pow(t, 6)) - (480 * pow(t, 4)) + (720 * pow(t, 2)) - 120) * exp(-pow(t, 2));
}

inline double deriative7(double t) {
    return ((-128 * pow(t, 7)) + (1344 * pow(t, 5)) - (3360 * pow(t, 3)) + (1680 * t)) * exp(-pow(t, 2));
}

inline double deriative8(double t) {
    return ((256 * pow(t, 8)) - (3584 * pow(t, 6)) + (13440 * pow(t, 4)) - (13440 * pow(t, 2)) + 1680) * exp(-pow(t, 2));
}

inline double deriative9(double t) {
    return ((-512 * pow(t, 9)) + (9216 * pow(t, 7)) - (48384 * pow(t, 5)) + (80640 * pow(t, 3)) - (30240 * t)) * exp(-pow(t, 2));
}

inline double deriative10(double t) {
    return ((1024 * pow(t, 10)) - (23040 * pow(t, 8)) + (161280 * pow(t, 6)) - (403200 * pow(t, 4)) + (302400 * pow(t, 2)) - 30240) * exp(-pow(t, 2));
}

inline double deriative11(double t) {
    return ((-2048 * pow(t, 11)) + (56320 * pow(t, 9)) - (506880 * pow(t, 7)) + (1774080 * pow(t, 5)) - (2217600 * pow(t, 3)) + (665280 * t)) * exp(-pow(t, 2));
}

inline double deriative12(double t) {
    return ((4096 * pow(t, 12)) - (135168 * pow(t, 10)) + (1520640 * pow(t, 8)) - (7096320 * pow(t, 6)) + (13305600 * pow(t, 4)) - (7983360 * pow(t, 2)) + 665280) * exp(-pow(t, 2));
}

inline double deriative13(double t) {
    return ((-8192 * pow(t, 13)) + (319488 * pow(t, 11)) - (4392960 * pow(t, 9)) + (26357760 * pow(t, 7)) - (69189120 * pow(t, 5)) + (69189120 * pow(t, 3)) - (17297280 * t)) * exp(-pow(t, 2));
}

inline double deriative14(double t) {
    return ((16384 * pow(t, 14)) - (745472 * pow(t, 12)) + (12300288 * pow(t, 10)) - (92252160 * pow(t, 8)) + (322882560 * pow(t, 6)) - (484323840 * pow(t, 4)) + (242161920 * pow(t, 2)) - 17297280) * exp(-pow(t, 2));
}

inline double deriative15(double t) {
    return ((-32768 * pow(t, 15)) + (1720320 * pow(t, 13)) - (33546240 * pow(t, 11)) + (307507200 * pow(t, 9)) - (1383782400 * pow(t, 7)) + (2905943040 * pow(t, 5)) - (2421619200 * pow(t, 3)) + (518918400 * t)) * exp(-pow(t, 2));
}

inline double deriative16(double t) {
    return ((65536 * pow(t, 16)) - (3932160 * pow(t, 14)) + (89456640 * pow(t, 12)) - (984023040 * pow(t, 10)) + (5535129600 * pow(t, 8)) - (15498362880 * pow(t, 6)) + (19372953600 * pow(t, 4)) - (8302694400 * pow(t, 2)) + 518918400) * exp(-pow(t, 2));
}

// pointer to deriative functions
inline double (*der_ptr[16])(double x) = {
    deriative1,
    deriative2,
    deriative3,
    deriative4,
    deriative5,
    deriative6,
    deriative7,
    deriative8,
    deriative9,
    deriative10,
    deriative11,
    deriative12,
    deriative13,
    deriative14,
    deriative15,
    deriative16,
};
}

namespace Functions {
// Hermite functions, returns -1 when n is smaller than 1
inline double h(unsigned int n, double t) {
    if (n >= 1)
        return exp(-pow(t, 2)) * pow(-1, n) * exp(pow(t, 2)) * (*Deriatives::der_ptr[n - 1])(t);
    else
        return -1;
}

// Calculates the factorial of x when it is greater than 1, otherwise returns -1
inline int fac(unsigned int x) {
    if (x == 0)
        return 1;
    if (x > 0) {
        int res = 1;
        for (size_t i = 1; i <= x; i++) {
            res = res * i;
        }
        return res;
    } else
        return -1;
}

// Calculates the Euclidean distance between two three-dimensional vectors
inline double euclidean_distance_3d(Vec3d a, Vec3d b) {
    return (a - b).calculate_p_norm(2);
}

// Kernel from Butz&Ooyen "A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization afterFocal Retinal Lesions"
// Calculates the attraction between two neurons, where a and b represent the position in three-dimensional space
inline double kernel(Vec3d a, Vec3d b) {
    return exp(-(pow(euclidean_distance_3d(a, b), 2) / pow(Constants::sigma, 2)));
}
}