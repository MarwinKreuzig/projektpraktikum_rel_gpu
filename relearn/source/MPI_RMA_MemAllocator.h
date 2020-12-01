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

#include "LogMessages.h"
#include "OctreeNode.h"

#include <mpi.h>

#include <algorithm>
#include <list>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

template<class T>
class MPI_RMA_MemAllocator {
public:
	MPI_RMA_MemAllocator() noexcept {

	}

	MPI_RMA_MemAllocator(const MPI_RMA_MemAllocator& other) = delete;
	MPI_RMA_MemAllocator(MPI_RMA_MemAllocator&& other) = delete;

	MPI_RMA_MemAllocator& operator=(const MPI_RMA_MemAllocator& other) = delete;
	MPI_RMA_MemAllocator& operator=(MPI_RMA_MemAllocator&& other) = delete;

	~MPI_RMA_MemAllocator() {

	}

	void set_size_requested(size_t size_requested) {
		this->size_requested = size_requested;
		this->avail_size = 0;

		// Number of objects "size_requested" Bytes correspond to
		max_num_objects = size_requested / sizeof(T);
		max_size = max_num_objects * sizeof(T);

		// Store size of MPI_COMM_WORLD
		MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

		base_ptr_offset = 0;
		avail_initialized = 0;
	}

	/**
	 * Memory allocation is not done in the constructor as
	 * the destructor would need to free it with MPI_Free_mem() later.
	 * Otherwise, it might happen that MPI_Finalize() is called before
	 * the destructor which causes an MPI error.
	 */
	void allocate_rma_mem() {
		// Allocate block of memory which is managed later on
		if (MPI_SUCCESS != MPI_Alloc_mem(max_size, MPI_INFO_NULL, &base_ptr)) {
			std::cout << __FUNCTION__ << ": MPI_Alloc_mem failed.";
			return;
		}
	}

	void init_free_object_list() {
		// Create free-memory list with one list element per object of type T
		for (size_t i = 0; i < max_num_objects; i++) {
			avail.push_back(base_ptr + base_ptr_offset + i);
			avail_size++;
		}
		avail_initialized = 1;
		min_num_avail_objects = avail_size;

		std::stringstream sstring;
		sstring << __FUNCTION__ << ": max_num_objects: " << max_num_objects << "  sizeof(OctreeNode): " << sizeof(OctreeNode);
		LogMessages::print_message_rank(sstring.str().c_str(), 0);
		sstring.str("");
	}

	void deallocate_rma_mem() {
		MPI_Free_mem(base_ptr);
	}

	// Create MPI RMA window with all the memory of the allocator
	// This call is collective over MPI_COMM_WORLD
	void create_rma_window() noexcept {
		// Set window's displacement unit
		displ_unit = 1;

		MPI_Win_create(base_ptr, max_size, displ_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &mpi_window);
	}

	// Free the MPI RMA window
	// This call is collective over MPI_COMM_WORLD
	void free_rma_window() {
		MPI_Win_free(&mpi_window);
	}

	// Store RMA window base pointers of all ranks
	void gather_rma_window_base_pointers() {
		// Vector must have space for one pointer from each rank
		base_pointers.resize(num_ranks);

		MPI_Allgather(&base_ptr, 1, MPI_AINT, base_pointers.data(), 1, MPI_AINT, MPI_COMM_WORLD);
	}

	// (i)   Allocate object of type T
	// (ii)  Call its constructor
	// (iii) Return pointer to object
	T* newObject() {
		T* ptr = nullptr;

		// No free objects available?
		if (avail.empty()) {
			std::cout << __FUNCTION__ << ": No free MPI-allocated memory available." << std::endl;
			ptr = nullptr;
		}
		else {
			ptr = avail.front();
			avail.pop_front();
			avail_size--;

			// Placement new operator
			// Only call constructor of object,
			// i.e. construct object at specified address "ptr"
			new (ptr) T;

			// Mark object as allocated (unavailable)
			unavail.insert(ptr);

			// Update min number of available objects
			min_num_avail_objects = std::min(avail_size, min_num_avail_objects);
		}
		return ptr;
	}

	// Call object's destructor and mark memory as available again
	void deleteObject(T* ptr) {
		// Check if object was allocated here
		if (unavail.erase(ptr)) {
			// Call destructor
			ptr->~T();

			// Mark object as free (available) again
			avail.push_front(ptr);
			avail_size++;
		}
		else {
			std::cout << __FUNCTION__ << ": Object's address unknown.\n";
		}
	}

	const MPI_Aint* get_base_pointers() const noexcept {
		return base_pointers.data();
	}

	// This can only be called before init_free_object_list()
	T* get_block_of_objects_memory(size_t num_objects) {
		T* ret = nullptr;

		if (avail_initialized) {
			std::cout << __FUNCTION__ << ": This function must not be called anymore as init_free_object_list()"
				" was called already." << std::endl;
			return ret;
		}

		if ((max_num_objects - num_objects) < 0) {
			std::cout << __FUNCTION__ << ": Not enough free MPI-allocated memory available." << std::endl;
			return ret;
		}

		max_num_objects -= num_objects;
		ret = base_ptr + base_ptr_offset;
		base_ptr_offset += num_objects;

		return ret;
	}

	size_t get_min_num_avail_objects() const noexcept {
		return min_num_avail_objects;
	}

	MPI_Win mpi_window;       // RMA window object
private:
	size_t size_requested;    // Bytes requested for the allocator
	size_t max_size;          // Size in Bytes of MPI-allocated memory
	size_t max_num_objects;   // Max number objects that are available
	T* base_ptr;              // Start address of MPI-allocated memory
	size_t base_ptr_offset;   // base_ptr + base_ptr_offset marks where free object list begins

	bool avail_initialized;   // List with free objects has been initialzed
	size_t min_num_avail_objects;  // Minimum number of objects available

	std::list<T*> avail;      // List of pointers to free memory blocks (each block of size sizeof(T))
	size_t avail_size;        // Size of avail. We don't use std::list.size() as some older stdlibc++ versions take O(n)
							  // and not O(1). This is a bug which we ran into on the Blue Gene/Q JUQUEEN
	std::set<T*> unavail;     // Set of pointers to used memory blocks (each block of size sizeof(T))
							  // A block can only be either in "avail" or "unavail". Blocks are moved between both.

	int num_ranks;                        // Number of ranks in MPI_COMM_WORLD
	int displ_unit;                       // RMA window displacement unit
	std::vector<MPI_Aint> base_pointers;  // RMA window base pointers of all procs
};
