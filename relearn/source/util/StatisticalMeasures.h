/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

/**
 * @brief This struct is used to aggregate different statistical parameters
 */
struct StatisticalMeasures {
    double min{ 0.0 };
    double max{ 0.0 };
    double avg{ 0.0 };
    double var{ 0.0 };
    double std{ 0.0 };
};

/**
 * @brief This enum provides a way to choose the attributes
 * of the neurons for which the statistics should be calculated
 */
enum class NeuronAttribute {
    Calcium,
    X,
    Fired,
    I_sync,
    Axons,
    AxonsConnected,
    DendritesExcitatory,
    DendritesExcitatoryConnected,
    DendritesInhibitory,
    DendritesInhibitoryConnected
};
