# Usage
The program offers multiple command line arguments. `--steps`, `--algorithm`, and one of`--num-neurons`, `--num-neurons-per-rank`, and `--file` are required, the rest is optional.

Command | Shortcut | Effect | Notes
--- | --- | --- | ---
`--algorithm <enum>` | `-a <enum>` | Chooses the algorithm that is used to find the target neurons, can be *naive*, *barnes-hut*, or *fast-multipole-method*
`--neuron-model <enum>`| | Chooses the neuron model, can be *poisson*, *izhikevich*, *aeif*, or *fitzhughnagumo* | 
`--num-neurons <size_t>` | `-n <size_t>` | Starts a simulation with approximately the given number of neurons | Excludes `--file` and `--num-neurons-per-rank` |
`--num-neurons-per-rank <size_t>` |  | Starts a simulation with the specified number of neurons per rank | Excludes `--file` and `--num-neurons` |
`--file <file-path>`| `-f <file-path>` | Starts a simulation with the neurons from the specified file | Excludes `--num-neurons` and `--num-neurons-per-rank`
`--graph <file-path>` | `-g <file-path>` | Starts a simulation with an initial connectivity from the specified file | Requires `--file`
`--enable-interrupts <file-path>` | | Specifies the enable interrupts for a simulation | 
`--disable-interrupts <file-path>` | | Specified the disable interrupts for a simulation |
`--creation-interrupts <file-path>` | | Specified the creation interrupts for a simulation |
`--log-prefix <str>` | `-p <str>` | Prefixes all log files with the given prefix |
`--log-path <dir-path>` | `-l <dir-path>` | Specified the directory in which the log files will be created |
`--steps <size_t>` | `-s <size_t>` | Specifies the number of steps in the simulation |
`--random-seed <uint32>` | `-r <uint32>` | The seed for the random number  generator |
`--openmp <int32>` | | Sets the number of OpenMP Threads | Must be greater than 0
`--theta <double>` | `-t <double>` | Sets the acceptance criterion for the Barnes-Hut-Algorithm | Must be from [0, 0.5], requires --algorithm barnes-hut
`--sigma <double>` | | Sets the scaling parameter for the probability kernel | Default is `750` 
`--interactive` | `-i` | Allows an interactive run (does not stop after the number of steps is completed) |
`--synaptic-elements-lower-bound <double>` | | Sets the lower bound of initial synaptic elements | Must be from [0, $\infty$)
`--synaptic-elements-upper-bound <double>` | | Sets the upper bound of initial synaptic elements | Must be from [0, $\infty$) and not less than the lower bound
`--target-ca <double>` | | Sets the target calcium for all neurons | Must be from [0, 100]. Default is `0.7`
`--initial-ca <double>`| | Sets the initial calcium for all neurons. | Default is `0.0`
`--beta <double>` | | Specifies the amount of calcium ions that are gathered whenever a neuron spikes |
`--calcium-decay` | | The decay constant of the intercellular calcium. | Default is `10000`
`--growth-rate <double>` | | Specifies the growth rate of the synaptic elements | Must be from [0, 1]
`--min-calcium-axons` | | Specified the minimum intercellular calcium for axons to grow. | Default is `0.4`
`--min-calcium-excitatory-dendrites` | | Specified the minimum intercellular calcium for excitatory dendrites to grow. | Default is `0.1`
`--min-calcium-inhibitory-dendrites` | | Specified the minimum intercellular calcium for inhibitory dendrites to grow. | Default is `0.0`
`--retract-ratio`| | The ratio by which vacant synapses retract. | Default is `0`
`--synapse_conductance <double>`| | The activity that is transfered when a connected neuron spikes. | Default is `0.03`
`--base-background-activity <double>` | | Specifies the base background activity by which all neurons are excited | Must be non-negative. The background activity is calculated as *base + N(mean, stddev)*
`--base-background-mean <double>` | | Specifies the mean background activity by which all neurons are excited | The background activity is calculated as *base + N(mean, stddev)*
`--base-background-stddev <double>` | | Specifies the standard deviation of the background activity by which all neurons are excited | Must be non-negative. The background activity is calculated as *base + N(mean, stddev)*

### In case you have compiled the program with MPI, you can also prefix the command with: `mpiexec -n 4 ...`
However, if you do so, the number of MPI processes must be a multiple of 2.t)
