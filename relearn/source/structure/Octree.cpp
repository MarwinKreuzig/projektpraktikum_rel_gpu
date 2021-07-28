/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

/*********************************************************************************
  * NOTE: We include Neurons.h here as the class Octree uses types from Neurons.h *
  * Neurons.h also includes Octree.h as it uses it too                            *
  *********************************************************************************/

#include "Octree.h"

#include "NodeCache.h"

/**
 * Do a postorder tree walk startring at "octree" and run the function "function" for every node when it is visited
 * Does ignore every node which's level in the octree is greater than "max_level"
 */


// Insert neuron into the tree

