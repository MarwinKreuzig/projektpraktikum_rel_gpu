#include "../googletest/include/gtest/gtest.h"
#include "../source/Config.h"
#include "../source/algorithm/FastMultipoleMethodsCell.h"
#include "../source/algorithm/FastMultipoleMethods.h"
#include "../source/structure/OctreeNode.h"
#include "../source/neurons/models/SynapticElements.h"
#include "../source/structure/Cell.h"
#include "../source/structure/Partition.h"

#include <stdio.h>
#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

using AdditionalCellAttributes = FastMultipoleMethodsCell;

const static double sigma = 750;

std::tuple<Vec3d, Vec3d> get_random_simulation_box_size_FMM(std::mt19937& mt) {
    std::uniform_real_distribution<double> urd(-10000.0, +10000.0);

    const auto rand_x_1 = urd(mt);
    const auto rand_x_2 = urd(mt);

    const auto rand_y_1 = urd(mt);
    const auto rand_y_2 = urd(mt);

    const auto rand_z_1 = urd(mt);
    const auto rand_z_2 = urd(mt);

    return {
        { std::min(rand_x_1, rand_x_2), std::min(rand_y_1, rand_y_2), std::min(rand_z_1, rand_z_2) },
        { std::max(rand_x_1, rand_x_2), std::max(rand_y_1, rand_y_2), std::max(rand_z_1, rand_z_2) }
    };
}

std::vector<std::tuple<Vec3d, size_t>> generate_random_neurons_FMM(const Vec3d& min, const Vec3d& max, size_t count, size_t max_id, std::mt19937& mt) {
    std::uniform_real_distribution<double> urd_x(min.get_x(), max.get_x());
    std::uniform_real_distribution<double> urd_y(min.get_y(), max.get_y());
    std::uniform_real_distribution<double> urd_z(min.get_z(), max.get_z());

    std::vector<size_t> ids(max_id);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), mt);

    std::vector<std::tuple<Vec3d, size_t>> return_value(count);

    for (auto i = 0; i < count; i++) {
        const auto rand_x = urd_x(mt);
        const auto rand_y = urd_y(mt);
        const auto rand_z = urd_z(mt);

        return_value[i] = { { rand_x, rand_y, rand_z }, ids[i] };
    }

    return return_value;
}

std::vector<std::tuple<Vec3d, size_t>> extract_neurons_FMM(OctreeNode<AdditionalCellAttributes>* root) {
    std::vector<std::tuple<Vec3d, size_t>> return_value;

    std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
    octree_nodes.push(root);

    while (!octree_nodes.empty()) {
        OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
        octree_nodes.pop();

        if (current_node->is_parent()) {
            const auto childs = current_node->get_children();
            for (auto i = 0; i < 8; i++) {
                const auto child = childs[i];
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        } else {
            const auto& cell = current_node->get_cell();
            const auto neuron_id = cell.get_neuron_id();
            const auto& opt_position = cell.get_dendrites_position();

            EXPECT_TRUE(opt_position.has_value());

            const auto position = opt_position.value();

            if (neuron_id < Constants::uninitialized) {
                return_value.emplace_back(position, neuron_id);
            }
        }
    }

    return return_value;
}

SynapticElements create_synaptic_elements_FMM(size_t size, std::mt19937& mt, double max_free, SignalType st) {
    SynapticElements se(ElementType::DENDRITE, 0.0);

    se.init(size);

    std::uniform_real_distribution<double> urd(0, max_free);

    for (auto i = 0; i < size; i++) {
        se.set_signal_type(i, st);
        se.update_count(i, urd(mt));
    }

    return se;
}

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

       const auto val_new0 = FastMultipoleMethods::function_derivative(-1, i + 1);
       const auto val_new1 = FastMultipoleMethods::function_derivative(0, i + 1);
       const auto val_new2 = FastMultipoleMethods::function_derivative(1, i + 1);

       EXPECT_NEAR(val_new0, res0, 0.01) << i;
       EXPECT_NEAR(val_new1, res1, 0.01) << i;
       EXPECT_NEAR(val_new2, res2, 0.01) << i;
   }
}

TEST(TestFastGauss, test_functions) {
   Vec3d a = { 0, 0, 0 };
   Vec3d b = { 0, 1, 0 };
   EXPECT_NEAR(FastMultipoleMethods::kernel(a, b, sigma), 0.999956, 0.0001);
   Vec3d c = { 0, 0, -1 };
   EXPECT_EQ(FastMultipoleMethods::kernel(a, a, sigma), 1);
   Vec3d e = { 6, 4.5, -3.4 };
   Vec3d f = { 0, -8.3, 2 };
   EXPECT_NEAR(FastMultipoleMethods::kernel(e, f, sigma), 0.9898, 0.01);
}

TEST(TestFastGauss, test_multiIndex) {
   EXPECT_EQ(FastMultipoleMethods::Multiindex::get_number_of_indices(), Constants::p3);

   const auto& indices = FastMultipoleMethods::Multiindex::get_indices();

   const std::array<unsigned int, 3> temp = indices[1];
   EXPECT_EQ(temp.at(0), 0);
   EXPECT_EQ(temp.at(1), 0);
   EXPECT_EQ(temp.at(2), 1);

   const std::array<unsigned int, 3> temp1 = indices[63];
   EXPECT_EQ(temp1.at(0), 3);
   EXPECT_EQ(temp1.at(1), 3);
   EXPECT_EQ(temp1.at(2), 3);
}