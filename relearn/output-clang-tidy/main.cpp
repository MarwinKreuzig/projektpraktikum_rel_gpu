/mnt/d/source/repos/relearn/relearn/source/algorithm/../structure/../neurons/Neurons.h:725:28: warning: non-const reference parameter 'pending_deletions', make it const or use a pointer [google-runtime-references]
        PendingDeletionsV& pending_deletions,
                           ^
/mnt/d/source/repos/relearn/relearn/source/algorithm/../structure/../neurons/Neurons.h:732:139: warning: non-const reference parameter 'pending_deletions', make it const or use a pointer [google-runtime-references]
    static void delete_synapses_process_requests(const MapSynapseDeletionRequests& synapse_deletion_requests_incoming, PendingDeletionsV& pending_deletions);
                                                                                                                                          ^
/mnt/d/source/repos/relearn/relearn/source/main.cpp:23:1: warning: #includes are not sorted properly [llvm-include-order]
#include "structure/Octree.h"
^        ~~~~~~~~~~~~~~~~~~~~
/mnt/d/source/repos/relearn/relearn/source/main.cpp:54:5: warning: an exception may be thrown in function 'main' which should not throw exceptions [bugprone-exception-escape]
int main(int argc, char** argv) {
    ^
/mnt/d/source/repos/relearn/relearn/source/main.cpp:93:11: warning: Value stored to 'opt_base_background_activity' during its initialization is never read [clang-analyzer-deadcode.DeadStores]
    auto* opt_base_background_activity = app.add_option("--base-background-activity", base_background_activity, "The base background activity by which all neurons are exited");
          ^
/mnt/d/source/repos/relearn/relearn/source/main.cpp:93:11: note: Value stored to 'opt_base_background_activity' during its initialization is never read
/mnt/d/source/repos/relearn/relearn/source/main.cpp:223:18: warning: Value stored to 'total_num_subdomains' during its initialization is never read [clang-analyzer-deadcode.DeadStores]
    const size_t total_num_subdomains = partition->get_total_num_subdomains();
                 ^
/mnt/d/source/repos/relearn/relearn/source/main.cpp:223:18: note: Value stored to 'total_num_subdomains' during its initialization is never read
/mnt/d/source/repos/relearn/relearn/source/main.cpp:323:33: warning: 6 is a magic number; consider replacing it with a named constant [cppcoreguidelines-avoid-magic-numbers]
    sim.register_neuron_monitor(6);
                                ^
/mnt/d/source/repos/relearn/relearn/source/main.cpp:323:33: warning: 6 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
/mnt/d/source/repos/relearn/relearn/source/main.cpp:324:33: warning: 1164 is a magic number; consider replacing it with a named constant [cppcoreguidelines-avoid-magic-numbers]
    sim.register_neuron_monitor(1164);
                                ^
/mnt/d/source/repos/relearn/relearn/source/main.cpp:324:33: warning: 1164 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
/mnt/d/source/repos/relearn/relearn/source/main.cpp:325:33: warning: 28001 is a magic number; consider replacing it with a named constant [cppcoreguidelines-avoid-magic-numbers]
    sim.register_neuron_monitor(28001);
                                ^
/mnt/d/source/repos/relearn/relearn/source/main.cpp:325:33: warning: 28001 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
/mnt/d/source/repos/relearn/relearn/source/main.cpp:343:22: warning: do not call c-style vararg functions [cppcoreguidelines-pro-type-vararg]
            auto n = scanf(" %c", &yn);
                     ^
/mnt/d/source/repos/relearn/relearn/source/main.cpp:343:22: warning: do not call c-style vararg functions [hicpp-vararg]
