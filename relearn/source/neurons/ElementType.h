#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <ostream>

/**
 * An instance of this enum classifies the synaptic elements of a neuron.
 * In this simulation, there exists exaclty two different ones: axonal elements and dendritic elements.
 * The distinction excitatory / inhibitory is made by the type SignalType.
 */
enum class ElementType { Axon,
    Dendrite };

inline std::ostream& operator<<(std::ostream& out, const ElementType& element_type) {
    if (element_type == ElementType::Axon) {
        return out << "Axon";
    } 

    if (element_type == ElementType::Dendrite) {
        return out << "Dendrite";
    }

    return out;
}
