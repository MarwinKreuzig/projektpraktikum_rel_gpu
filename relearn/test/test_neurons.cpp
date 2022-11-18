#include "gtest/gtest.h"

#include "neurons/CalciumCalculator.h"
#include "neurons/Neurons.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticElements.h"
#include "structure/Partition.h"

#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticElements.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/Neurons.h"
#include "structure/Partition.h"

TEST_F(NeuronsTest, testNeuronsConstructor) {
    auto partition = std::make_shared<Partition>(1, 0);

    auto model = std::make_unique<models::PoissonModel>();
    auto calcium = std::make_unique<CalciumCalculator>();
    auto dends_ex = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto dends_in = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto axs = std::make_unique<SynapticElements>(ElementType::Axon, 0.2);

    Neurons neurons{ partition, std::move(model), std::move(calcium), std::move(axs), std::move(dends_ex), std::move(dends_in) };
}
