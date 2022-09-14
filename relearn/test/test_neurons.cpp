#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/neurons/CalciumCalculator.h"
#include "../source/neurons/Neurons.h"
#include "../source/neurons/models/NeuronModels.h"
#include "../source/neurons/models/SynapticElements.h"
#include "../source/structure/Partition.h"

TEST_F(NeuronsTest, testNeuronsConstructor) {
    auto partition = std::make_shared<Partition>(1, 0);

    auto model = std::make_unique<models::PoissonModel>();
    auto calcium = std::make_unique<CalciumCalculator>();
    auto dends_ex = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto dends_in = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto axs = std::make_unique<SynapticElements>(ElementType::Axon, 0.2);

    Neurons neurons{ partition, std::move(model), std::move(calcium), std::move(axs), std::move(dends_ex), std::move(dends_in) };
}
