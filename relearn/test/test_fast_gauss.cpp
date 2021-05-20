#include "../googletest/include/gtest/gtest.h"

#include "../source/Config.h"
#include "../source/DeriativesAndFunctions.h"
#include "../source/Cell.h"
#include "../source/OctreeNode.h"
#include "../source/Multiindex.h"
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

    for (int i = 0; i < 16; i++) {
        EXPECT_NEAR((*Deriatives::der_ptr[i+1])(-1), result[(i * 3) + 0], 0.01);
        EXPECT_NEAR((*Deriatives::der_ptr[i+1])(0), result[(i * 3) + 1], 0.01);
        EXPECT_NEAR((*Deriatives::der_ptr[i+1])(1), result[(i * 3) + 2], 0.01);
    }
}

TEST(TestFastGauss, test_functions) {

    Vec3d a = { 0, 0, 0 };
    Vec3d b = { 0, 1, 0 };
    EXPECT_EQ(Functions::euclidean_distance_3d(a, b), 1);
    EXPECT_NEAR(Functions::kernel(a, b), 0.999956, 0.0001);
    Vec3d c = { 0, 0, -1 };
    EXPECT_EQ(Functions::euclidean_distance_3d(a, c), 1);
    EXPECT_EQ(Functions::euclidean_distance_3d(a, a), 0);
    EXPECT_EQ(Functions::kernel(a, a), 1);
    Vec3d e = { 6, 4.5, -3.4 };
    Vec3d f = { 0, -8.3, 2 };
    EXPECT_NEAR(Functions::euclidean_distance_3d(e, f), 15.132745950421556, 0.01);
    EXPECT_NEAR(Functions::kernel(e, f), 0.9898, 0.01);
}

TEST(TestFastGauss, test_calc_attractiveness) {
    std::vector<Vec3d> test_source;
    test_source.reserve(5);
    std::vector<Vec3d> test_target;
    test_target.reserve(5);

    test_source.push_back({  1,  1,  1 });
    test_source.push_back({  1, 10,  1 });
    test_source.push_back({ 10, 10,  1 });
    test_source.push_back({  1,  1, 10 });
    test_source.push_back({ 10, 10,  1 });

    test_target.push_back({ 201, 201, 201 });
    test_target.push_back({ 201, 210, 201 });
    test_target.push_back({ 210, 201, 201 });
    test_target.push_back({ 201, 201, 210 });
    test_target.push_back({ 210, 210, 201 });

    Vec3d source_center = { 4.6, 6.4, 2.8 };
    Vec3d target_center = { 204.6, 206.4, 202.8 };


    double gauss_res = Functions::calc_direct_gauss(&test_source, &test_target);
    double taylor_res = Functions::calc_taylor_expansion(&test_source,&test_target, &target_center);
    printf("Gauss result = %f \n", gauss_res);
    printf("Taylor result = %f \n", taylor_res);
    std::vector<double> hermite_coef;
    hermite_coef.reserve(64);
    Functions::calc_hermite_coefficients(&source_center, &test_source, &hermite_coef);
    double hermite_res = Functions::calc_hermite(&test_target, &hermite_coef, &source_center);
    printf("Hermite result = %f \n", hermite_res);

    EXPECT_NEAR(gauss_res, taylor_res, 0.01);
    EXPECT_NEAR(gauss_res, pow(test_source.size(),2) * Functions::kernel(source_center, target_center), 0.01);
    EXPECT_NEAR(gauss_res, hermite_res, 0.01);
}

TEST(TestFastGauss, test_interaction_list) {
    OctreeNode test_node;
    OctreeNode list_node;
    OctreeNode* temp;

    list_node.set_cell_neuron_id(1235);

    EXPECT_EQ(test_node.get_from_interactionlist(1),nullptr);
    test_node.add_to_interactionlist(&list_node);
    EXPECT_EQ(test_node.get_interactionlist_length(),1);
    temp = test_node.get_from_interactionlist(0);
    EXPECT_EQ(temp->get_cell().get_neuron_id(), 1235);
}

TEST(TestFastGauss, test_multiIndex) {
Multiindex m = Multiindex();
EXPECT_EQ(m.get_number_of_indices(), 64);

std::array<int,3>* temp = m.get_indice_at(1);
EXPECT_EQ(temp->at(0),0);
EXPECT_EQ(temp->at(1),0);
EXPECT_EQ(temp->at(2),1);

temp = m.get_indice_at(63);
EXPECT_EQ(temp->at(0),3);
EXPECT_EQ(temp->at(1),3);
EXPECT_EQ(temp->at(2),3);

}