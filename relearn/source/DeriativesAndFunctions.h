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
#include "Multiindex.h"

namespace Deriatives {

inline double original_func(double t){
    return exp(-pow(t, 2));
}   
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
inline double (*der_ptr[17])(double x) = {
    original_func,
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
    if (n >= 0)
        return exp(-pow(t, 2)) * pow(-1, n) * exp(pow(t, 2)) * (*Deriatives::der_ptr[n])(t);
    else
        return -1;
}

inline double h_multiindex(std::array<int, 3>* n, Vec3d* t) {
    return h(n->at(0), t->get_x()) * h(n->at(1), t->get_y()) * h(n->at(2), t->get_z());
}

// Calculates the factorial of a multiindex x
inline int fac_multiindex(std::array<int, 3>* x) {
    int temp;
    int result = 1;
    for (int i = 0; i < 3; i++) {
        temp = 1;
        for (int j = 1; j < x->at(i); j++) {
            temp = temp * j;
        }
        result = result * temp;
    }
    return result;
}

inline double pow_multiindex(Vec3d* base_vector, std::array<int, 3>* exponent) {
    return pow(base_vector->get_x(), exponent->at(0)) * pow(base_vector->get_y(), exponent->at(1)) * pow(base_vector->get_z(), exponent->at(2));
}
inline double abs_multiindex(std::array<int, 3>* x) {
    return x->at(0) + x->at(1) + x->at(2);
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

inline double calc_taylor_expansion(std::vector<Vec3d>* sources, std::vector<Vec3d>* targets, Vec3d* center_target) {
    Multiindex m = Multiindex();
    int num_coef = m.get_number_of_indices();
    std::vector<double> taylor_coef;
    taylor_coef.reserve(64);
    double result = 0;
    double temp;
    Vec3d temp_vec = Vec3d();
    std::array<int,3>* current_index;

    for (size_t b = 0; b < num_coef; b++) {
        temp = 0;
        current_index = m.get_indice_at(b);
        for (size_t j = 0; j < sources->size(); j++) {
            temp_vec.set_x((sources->at(j).get_x()-center_target->get_x())/Constants::sigma);
            temp_vec.set_y((sources->at(j).get_y()-center_target->get_y())/Constants::sigma);
            temp_vec.set_z((sources->at(j).get_z()-center_target->get_z())/Constants::sigma);
            temp += h_multiindex(current_index, &temp_vec);
        }
        double C = (pow(-1, abs_multiindex(current_index)) / Functions::fac_multiindex(current_index)) * temp;
        //printf("taylor coef at %i = %f \n",b,C);
        taylor_coef.push_back(C);
    }
    //Evaluate Taylor series at all sources
    for (unsigned int j = 0; j < sources->size(); j++) {
        temp = 0;
        temp_vec.set_x((targets->at(j).get_x() - center_target->get_x()) / Constants::sigma);
        temp_vec.set_y((targets->at(j).get_y() - center_target->get_y()) / Constants::sigma);
        temp_vec.set_z((targets->at(j).get_z() - center_target->get_z()) / Constants::sigma);
        for (size_t b = 0; b < num_coef; b++) {
            temp += taylor_coef.at(b) * pow_multiindex(&temp_vec, m.get_indice_at(b));
        }
        result += temp;
    }
    return result;
}

//calculates direct gauss of two vectors with 3D positions
inline double calc_direct_gauss(std::vector<Vec3d>* sources, std::vector<Vec3d>* targets) {
    double result = 0;
    for (int t = 0; t < targets->size(); t++) {
        for (size_t s = 0; s < sources->size(); s++) {
            result += Functions::kernel(targets->at(t), sources->at(s));
        }
    }
    return result;
}

inline void calc_hermite_coefficients(Vec3d* center_of_source, std::vector<Vec3d>* sources, std::vector<double>* coef) {
    Multiindex m = Multiindex();
    int num_coef = m.get_number_of_indices();
    double temp;
    Vec3d temp_vec = Vec3d();
    for (size_t a = 0; a < num_coef; a++) {
        temp = 0;
        for (size_t i = 0; i < sources->size(); i++) {
            temp_vec.set_x((sources->at(i).get_x() - center_of_source->get_x()) / Constants::sigma);
            temp_vec.set_y((sources->at(i).get_y() - center_of_source->get_y()) / Constants::sigma);
            temp_vec.set_z((sources->at(i).get_z() - center_of_source->get_z()) / Constants::sigma);
            temp += pow_multiindex(&temp_vec, m.get_indice_at(a));
        }
        //printf("temp  = %f \n", temp);
        coef->push_back((1 / fac_multiindex(m.get_indice_at(a))) * temp);
        //printf("hermite coef an Stelle %i = %f \n", a, coef->at(a-1));
    }
}

inline double calc_hermite(std::vector<Vec3d>* targets, std::vector<double>* coef, Vec3d* center_of_sources) {
    double result = 0;
    Multiindex m = Multiindex();
    int coef_num = m.get_number_of_indices();
    double temp;
    Vec3d temp_vec = Vec3d();

    for (size_t j = 0; j < targets->size(); j++) {
        temp = 0;
        temp_vec.set_x((targets->at(j).get_x() - center_of_sources->get_x()) / Constants::sigma);
        temp_vec.set_y((targets->at(j).get_y() - center_of_sources->get_y()) / Constants::sigma);
        temp_vec.set_z((targets->at(j).get_z() - center_of_sources->get_z()) / Constants::sigma); 
        for (size_t a = 0; a < coef_num; a++) {
            //printf("hermite coef an Stelle %i = %f \n", a, coef->at(a-1));
            temp += coef->at(a)* h_multiindex(m.get_indice_at(a),&temp_vec);
        }
        result += temp;
    }
    return result;
}
}