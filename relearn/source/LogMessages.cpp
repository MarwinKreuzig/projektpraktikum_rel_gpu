#include <iostream>

#include "LogMessages.h"
#include "MPIInfos.h"


void LogMessages::print_message(char const* string) {
	std::cout << "[INFO]  " << string << "\n";
}

// Print tagged message only at MPI rank "rank"
void LogMessages::print_message_rank(char const* string, int rank) {
	if (rank == MPIInfos::my_rank || rank == -1) {
		std::cout << "[INFO:Rank " << MPIInfos::my_rank << "]  " << string << "\n";
	}
}

void LogMessages::print_error(char const* string) {
	std::cout << "[ERROR]  " << string << "\n";
}

void LogMessages::print_debug(char const* string) {
	std::cout << "[DEBUG]  " << string << "\n";
}
