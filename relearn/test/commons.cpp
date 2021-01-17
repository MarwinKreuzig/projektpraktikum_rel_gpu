#include "commons.h"

#include "../source/MPIWrapper.h"
#include "../source/RelearnException.h"

std::mt19937 mt;


void setup() {
    MPIWrapper::init_globals();
	
	RelearnException::hide_messages = true;
}



