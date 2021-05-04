#include "commons.h"

#include "../source/io/LogFiles.h"
#include "../source/mpi/MPIWrapper.h"
#include "../source/util/RelearnException.h"

#include <mutex>

std::mt19937 mt;
std::once_flag some_flag;

void setup() {

    auto lambda = []() {
        char* argument = (char*)"./runTests";
        MPIWrapper::init(1, &argument);
        MPIWrapper::init_buffer_octree(1);
        LogFiles::init();
    };

    std::call_once(some_flag, lambda);

    RelearnException::hide_messages = true;
}
