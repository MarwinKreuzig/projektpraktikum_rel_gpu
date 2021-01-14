#include "commons.h"

#include "../source/MPIWrapper.h"
#include "../source/RelearnException.h"

std::mt19937 mt;


void setup() {
	MPIWrapper::num_ranks = 1;
	MPIWrapper::my_rank = 0;
	
	RelearnException::hide_messages = true;
}



