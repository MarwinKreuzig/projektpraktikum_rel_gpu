#include "../googletest/include/gtest/gtest.h"

#include "../source/Config.h"
#include "../source/Deriatives.h"
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
        -606.27, 1680, -606.27
    };

for(int i = 0; i < 8; i++){
    EXPECT_NEAR((*der_ptr[i])(-1), result[(i*3)+0], 0.01);
    EXPECT_NEAR((*der_ptr[i])(0), result[(i*3)+1], 0.01);
    EXPECT_NEAR((*der_ptr[i])(1), result[(i*3)+2], 0.01);
    }   
}

TEST(TestFastGauss, test_coefficients) {
    Cell c;
    c.init_coefficients();
    for (size_t i = 0; i < Constants::coefficient_num; i++)
    {
        EXPECT_EQ(c.get_coefficient(i),0);
    }

    //generate Input
    double p[Constants::coefficient_num];
    for (size_t i = 0; i < Constants::coefficient_num; i++)
    {
        p[i]=1;
    }

    c.add_to_coefficients(p);
    for (size_t i = 0; i < Constants::coefficient_num; i++)
    {
        EXPECT_EQ(c.get_coefficient(i),1);
    }
}

