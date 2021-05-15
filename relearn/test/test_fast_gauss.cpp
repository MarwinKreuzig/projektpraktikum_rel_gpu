#include "../googletest/include/gtest/gtest.h"

#include "../source/Config.h"
#include "../source/DeriativesAndFunctions.h"
#include "../source/Cell.h"
#include <stdio.h>

TEST(TestFastGauss, test_deriatives) {
   double result[] = {
        0.74, 0, -0.74,
        0.74, -2, 0.74,
        -1.47, 0, 1.47,
        -7.36, 12, -7.36,
        -2.94, 0, 2.94,
        67.69, -120, 67.69,
        170.7, 0, -170.7,
        -606.27, 1680, -606.27,
        -3943.67, 0, 3943.67,
        3025.44, -30240, 3025.44,
        84924.23, 0, -84924.23,
        103288.77, 665280, 103288.77,
        -1831604.05, 0, 1831604.05,
        -6348716.24, -17297280, -6348716.24,
        38587480.85, 0, -38587480.85,
        267636449.02, 518918400, 267636449.02
    };

for(int i = 0; i < 16; i++){
    printf("%i Ableitung \n",i+1);
    EXPECT_NEAR((*Deriatives::der_ptr[i])(-1), result[(i*3)+0], 0.01);
    EXPECT_NEAR((*Deriatives::der_ptr[i])(0), result[(i*3)+1], 0.01);
    EXPECT_NEAR((*Deriatives::der_ptr[i])(1), result[(i*3)+2], 0.01);
    }   
}

TEST(TestFastGauss, test_functions) {

    int result[] = { 1, 2, 6, 24, 120, 720, 5040, 40320 };
    for (size_t i = 1; i < 9; i++) {
        EXPECT_EQ(Functions::fac(i), result[i-1]);
    }

    Vec3d a = {0,0,0};
    Vec3d b = {0,1,0};
    EXPECT_EQ(Functions::euclidean_distance_3d(a,b),1);
    EXPECT_NEAR(Functions::kernel(a,b), 0.999956, 0.0001);
    Vec3d c = {0,0,-1};
    EXPECT_EQ(Functions::euclidean_distance_3d(a,c),1);
    EXPECT_EQ(Functions::euclidean_distance_3d(a,a),0);
    EXPECT_EQ(Functions::kernel(a,a), 1);
    Vec3d e = {6, 4.5, -3.4};
    Vec3d f = {0,-8.3,2};
    EXPECT_NEAR(Functions::euclidean_distance_3d(e,f),15.132745950421556,0.01);
    EXPECT_NEAR(Functions::kernel(e,f), 0.9898, 0.01);

    EXPECT_EQ(Functions::h(0,5),-1);
    EXPECT_EQ(Functions::h(1,0),0);
    EXPECT_NEAR(Functions::h(1,1), 0.735759, 0.01);
    EXPECT_NEAR(Functions::h(1,-1), -0.735759, 0.01);
    EXPECT_NEAR(Functions::h(2,0),-2, 0.01);
    EXPECT_NEAR(Functions::h(2,1), 0.735759, 0.01);
    EXPECT_NEAR(Functions::h(2,-1), 0.735759, 0.01);
}
