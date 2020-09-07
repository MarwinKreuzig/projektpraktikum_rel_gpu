/*
 * File:   LogMessages.h
 * Author: rinke
 *
 * Created on Apr 13, 2016
 */

#ifndef LOGMESSAGES_H
#define LOGMESSAGES_H

#include "MPIInfos.h"
#include <iostream>

class LogMessages {
public:

	/**
	 * Static functions for printing a tagged log message to std::cout
	 */
	static void print_message(char const* string) {
		std::cout << "[INFO]  " << string << "\n";
	}

	// Print tagged message only at MPI rank "rank"
	static void print_message_rank(char const* string, int rank) {
		if (rank == MPIInfos::my_rank || rank == -1) {
			std::cout << "[INFO:Rank " << MPIInfos::my_rank << "]  " << string << "\n";
		}
	}

	static void print_error(char const* string) {
		std::cout << "[ERROR]  " << string << "\n";
	}

	static void print_debug(char const* string) {
		std::cout << "[DEBUG]  " << string << "\n";
	}
};

#endif /* LOGMESSAGES_H */
