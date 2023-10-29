/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_memory_footprint.h"

#include "mpi/MPIWrapper.h"
#include "util/MemoryFootprint.h"

#include <iostream>
#include <vector>

TEST_F(MemoryFootprintTest, testInsert) {
    if (MPIWrapper::get_number_ranks() != 1) {
        if (MPIWrapper::get_my_rank() == MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    MemoryFootprint mf{ 2 };
    mf.emplace("Hello", 100);
    mf.emplace("Some size", 200);
}

TEST_F(MemoryFootprintTest, testInsertTypes) {
    if (MPIWrapper::get_number_ranks() != 1) {
        if (MPIWrapper::get_my_rank() == MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    std::vector<char> buffer{};
    buffer.push_back('c');
    buffer.push_back('h');
    buffer.push_back('a');
    buffer.push_back('r');
    buffer.push_back('\0');

    std::vector<char> constant_buffer{};
    constant_buffer.push_back('c');
    constant_buffer.push_back('o');
    constant_buffer.push_back('n');
    constant_buffer.push_back('s');
    constant_buffer.push_back('t');
    constant_buffer.push_back('\0');

    char* mem = buffer.data();
    const char* const_mem = constant_buffer.data();

    std::string constant_string{ "Constant string" };

    MemoryFootprint mf{ 5 };
    mf.emplace("Hello", 100);
    mf.emplace(mem, 200);
    mf.emplace(const_mem, 300);
    mf.emplace(std::string("Moveable string"), 400);
    mf.emplace(constant_string, 500);
}

TEST_F(MemoryFootprintTest, testRetrieveTypes) {
    if (MPIWrapper::get_number_ranks() != 1) {
        if (MPIWrapper::get_my_rank() == MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    std::vector<char> buffer{};
    buffer.push_back('c');
    buffer.push_back('h');
    buffer.push_back('a');
    buffer.push_back('r');
    buffer.push_back('\0');

    std::vector<char> constant_buffer{};
    constant_buffer.push_back('c');
    constant_buffer.push_back('o');
    constant_buffer.push_back('n');
    constant_buffer.push_back('s');
    constant_buffer.push_back('t');
    constant_buffer.push_back('\0');

    char* mem = buffer.data();
    const char* const_mem = constant_buffer.data();

    std::string constant_string{ "Constant string" };

    MemoryFootprint mf{ 5 };
    mf.emplace("Hello", 100);
    mf.emplace(mem, 200);
    mf.emplace(const_mem, 300);
    mf.emplace(std::string("Moveable string"), 400);
    mf.emplace(constant_string, 500);

    const auto& dict = mf.get_descriptions();

    ASSERT_EQ(dict.size(), 5);

    ASSERT_TRUE(dict.contains("char"));
    ASSERT_TRUE(dict.contains("const"));
    ASSERT_TRUE(dict.contains("Constant string"));
    ASSERT_TRUE(dict.contains("Moveable string"));
    ASSERT_TRUE(dict.contains("Hello"));

    ASSERT_EQ(dict.at("Hello"), 100);
    ASSERT_EQ(dict.at("char"), 200);
    ASSERT_EQ(dict.at("const"), 300);
    ASSERT_EQ(dict.at("Moveable string"), 400);
    ASSERT_EQ(dict.at("Constant string"), 500);
}

TEST_F(MemoryFootprintTest, testRetrieve) {
    if (MPIWrapper::get_number_ranks() != 1) {
        if (MPIWrapper::get_my_rank() == MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    MemoryFootprint mf{ 5 };
    mf.emplace("Hello 1", 4);
    mf.emplace("Hello 2", 4);
    mf.emplace("Hello 3", 4);
    mf.emplace("Hello 4", 4);
    mf.emplace("Hello 5", 4);
    mf.emplace("Hello 6", 4);
    mf.emplace("Hello 7", 4);
    mf.emplace("Hello 8", 4);

    const auto& dict = mf.get_descriptions();

    ASSERT_EQ(dict.size(), 8);

    ASSERT_EQ(dict.at("Hello 1"), 4);
    ASSERT_EQ(dict.at("Hello 2"), 4);
    ASSERT_EQ(dict.at("Hello 3"), 4);
    ASSERT_EQ(dict.at("Hello 4"), 4);
    ASSERT_EQ(dict.at("Hello 5"), 4);
    ASSERT_EQ(dict.at("Hello 6"), 4);
    ASSERT_EQ(dict.at("Hello 7"), 4);
    ASSERT_EQ(dict.at("Hello 8"), 4);
}

TEST_F(MemoryFootprintTest, testOverwrite) {
    if (MPIWrapper::get_number_ranks() != 1) {
        if (MPIWrapper::get_my_rank() == MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    MemoryFootprint mf{ 5 };
    mf.emplace("Hello 1", 4);
    mf.emplace("Hello 2", 4);
    mf.emplace("Hello 3", 4);
    mf.emplace("Hello 4", 4);
    mf.emplace("Hello 5", 4);
    mf.emplace("Hello 6", 4);
    mf.emplace("Hello 7", 4);
    mf.emplace("Hello 8", 4);

    mf.emplace("Hello 1", 2);
    mf.emplace("Hello 2", 2);
    mf.emplace("Hello 3", 2);
    mf.emplace("Hello 4", 4);
    mf.emplace("Hello 5", 4);
    mf.emplace("Hello 6", 7);
    mf.emplace("Hello 7", 7);
    mf.emplace("Hello 8", 7);

    const auto& dict = mf.get_descriptions();

    ASSERT_EQ(dict.size(), 8);

    ASSERT_EQ(dict.at("Hello 1"), 4);
    ASSERT_EQ(dict.at("Hello 2"), 4);
    ASSERT_EQ(dict.at("Hello 3"), 4);
    ASSERT_EQ(dict.at("Hello 4"), 4);
    ASSERT_EQ(dict.at("Hello 5"), 4);
    ASSERT_EQ(dict.at("Hello 6"), 4);
    ASSERT_EQ(dict.at("Hello 7"), 4);
    ASSERT_EQ(dict.at("Hello 8"), 4);
}
