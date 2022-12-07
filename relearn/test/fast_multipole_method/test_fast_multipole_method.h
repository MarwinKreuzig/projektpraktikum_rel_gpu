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

// class FMMTest : public RelearnTest {
// protected:
//     static void SetUpTestSuite() {
//         SetUpTestCaseTemplate<FastMultipoleMethodsCell>();
//     }
//   Stack<FastMultipoleMethodsBase::stack_entry> init_stack(FastMultipoleMethods fmm, const SignalType signal_type_needed) { return FastMultipoleMethodsBase::init_stack(signal_type_needed); }
//
//   void unpack_node_pair(FastMultipoleMethods fmm, Stack<FastMultipoleMethodsBase::stack_entry>& stack) { return FastMultipoleMethodsBase::unpack_node_pair(stack); }
//
//   FastMultipoleMethodsBase::interaction_list_type align_interaction_list(FastMultipoleMethods fmm, OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>* source_node, OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>* target_parent, const SignalType signal_type) { return fmm.align_interaction_list(source_node, target_parent, signal_type); }
//
// std::array<double, Constants::p3> calc_hermite_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, double sigma, SignalType signal_type_needed) { return FastMultipoleMethods::calc_hermite_coefficients(source, sigma, signal_type_needed); }
// CalculationType check_calculation_requirements(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, double sigma, SignalType signal_type_needed){return FastMultipoleMethods::check_calculation_requirements(source, target, sigma, signal_type_needed);}
// const std::vector<std::pair<FastMultipoleMethods::position_type, FastMultipoleMethods::counter_type>> get_all_positions_for(OctreeNode<FastMultipoleMethodsCell>* node, const ElementType type, const SignalType signal_type_needed){return FastMultipoleMethodsBase<FastMultipoleMethodsCell>::get_all_positions_for(node, type, signal_type_needed);}
// double calc_taylor(const OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, double sigma, SignalType signal_type_needed){return FastMultipoleMethods::calc_taylor(source, target, sigma, signal_type_needed);}
// double calc_direct_gauss(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, double sigma, SignalType signal_type_needed){return FastMultipoleMethods::calc_direct_gauss(source, target, sigma, signal_type_needed);}
// double calc_hermite(const OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, const std::array<double, Constants::p3>& coefficients_buffer, double sigma, SignalType signal_type_needed){return FastMultipoleMethods::calc_hermite(source, target, coefficients_buffer, sigma, signal_type_needed);}
//};
