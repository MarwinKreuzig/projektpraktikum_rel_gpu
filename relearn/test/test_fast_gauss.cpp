#include "../googletest/include/gtest/gtest.h"
#include "../source/Config.h"
#include "../source/util/DeriativesAndFunctions.h"
#include "../source/structure/Cell.h"
#include "../source/structure/OctreeNode.h"
#include "../source/util/Multiindex.h"
#include <stdio.h>
const static double sigma = 750;

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
        const auto res0 = result[(i * 3) + 0];
        const auto res1 = result[(i * 3) + 1];
        const auto res2 = result[(i * 3) + 2];

        const auto val_new0 = Deriatives::function_derivative(-1, i + 1);
        const auto val_new1 = Deriatives::function_derivative(0, i + 1);
        const auto val_new2 = Deriatives::function_derivative(1, i + 1);

        EXPECT_NEAR(val_new0, res0, 0.01) << i;
        EXPECT_NEAR(val_new1, res1, 0.01) << i;
        EXPECT_NEAR(val_new2, res2, 0.01) << i;
    }
}

TEST(TestFastGauss, test_functions) {
    Vec3d a = { 0, 0, 0 };
    Vec3d b = { 0, 1, 0 };
    EXPECT_NEAR(Functions::kernel(a, b, sigma), 0.999956, 0.0001);
    Vec3d c = { 0, 0, -1 };
    EXPECT_EQ(Functions::kernel(a, a, sigma), 1);
    Vec3d e = { 6, 4.5, -3.4 };
    Vec3d f = { 0, -8.3, 2 };
    EXPECT_NEAR(Functions::kernel(e, f, sigma), 0.9898, 0.01);
}

TEST(TestFastGauss, test_interaction_list) {
    OctreeNode test_node;
    OctreeNode list_node;
    const OctreeNode* temp;

    list_node.set_cell_neuron_id(1235);

    EXPECT_EQ(test_node.get_from_interactionlist(1), nullptr);
    test_node.add_to_interactionlist(&list_node);
    EXPECT_EQ(test_node.get_interactionlist_length(), 1);
    temp = test_node.get_from_interactionlist(0);
    EXPECT_EQ(temp->get_cell().get_neuron_id(), 1235);
}

TEST(TestFastGauss, test_multiIndex) {
    EXPECT_EQ(Multiindex::get_number_of_indices(), Constants::p3);

    const auto& indices = Multiindex::get_indices();

    const std::array<unsigned int, 3> temp = indices[1];
    EXPECT_EQ(temp.at(0), 0);
    EXPECT_EQ(temp.at(1), 0);
    EXPECT_EQ(temp.at(2), 1);

    const std::array<unsigned int, 3> temp1 = indices[63];
    EXPECT_EQ(temp1.at(0), 3);
    EXPECT_EQ(temp1.at(1), 3);
    EXPECT_EQ(temp1.at(2), 3);
}