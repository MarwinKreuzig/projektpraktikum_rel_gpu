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
//e
const double E = exp(1.);

//this file contains all x derivatives of e that are required for the Fast Gauss algorithm 
double deriative1(int t) {
    return -2 * t * pow(E, -pow(t, 2));
}

double deriative2(int t) {
    return ((4 * pow(t, 2)) - 2) * pow(E, -pow(t, 2));
}

double deriative3(int t) {
    return -((8 * pow(t, 3)) - (12 * t)) * pow(E, -pow(t, 2));
}

double deriative4(int t) {
    return ((16 * pow(t, 4)) - (48 * pow(t, 2)) + 12) * pow(E, -pow(t, 2));
}

double deriative5(int t) {
    return -((32 * pow(t, 5)) - (160 * pow(t, 3)) + (120 * t)) * pow(E, -pow(t, 2));
}

double deriative6(int t) {
    return ((64 * pow(t, 6)) - (480 * pow(t, 4)) + (720 * pow(t, 2)) - 120) * pow(E, -pow(t, 2));
}

double deriative7(int t) {
    return ((-128 * pow(t, 7)) + (1344 * pow(t, 5)) - (3360 * pow(t, 3)) + (1680 * t)) * pow(E, -pow(t, 2));
}

double deriative8(int t) {
    return ((256 * pow(t, 8)) - (3584 * pow(t, 6)) + (13440 * pow(t, 4)) - (13440 * pow(t, 2)) + 1680) * pow(E, -pow(t, 2));
}

double (*der_ptr[8])(int x) = {
    deriative1,
    deriative2,
    deriative3,
    deriative4,
    deriative5,
    deriative6,
    deriative7,
    deriative8
};

double h(int n, double t){
    return pow(E,-pow(t,2))*pow(-1,n)*pow(E,pow(t,2))*(*der_ptr[n-1])(t);
}

int fac(int x){
    int res=0;
    for (size_t i = 1; i <= x; i++)
    {
        res+=i;
    }
    return res;
}

double euclidean_distance_3d (Vec3d a, Vec3d b){
    return sqrt( pow(a.get_x()-b.get_x(),2) + pow(a.get_y()-b.get_y(),2) +pow(a.get_z()-b.get_z(),2)  );
}

double kernel (Vec3d a, Vec3d b){
    return pow(E,-(pow(euclidean_distance_3d(a,b),2)/pow(Constants::sigma,2)));
}