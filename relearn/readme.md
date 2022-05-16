
# Usage
The program offers multiple command line arguments. `--steps`, `--algorithm`, and one of`--num-neurons`, `--num-neurons-per-rank`, and `--file` are required, the rest is optional.

Command | Shortcut | Effect | Notes
--- | --- | --- | ---
`--steps <size_t>` | `-s <size_t>` | Specifies the number of steps in the simulation |
`--first-plasticity-step <size_t>` | | Specifies the first update step in which the plasticity is updated |
`--interactive` | `-i` | Allows an interactive run (does not stop after the number of steps is completed) |
`--random-seed <uint32>` | `-r <uint32>` | The seed for the random number  generator |
`--openmp <int32>` | | Sets the number of OpenMP Threads | Must be greater than 0 
`--log-path <dir-path>` | `-l <dir-path>` | Specified the directory in which the log files will be created |
`--log-prefix <str>` | `-p <str>` | Prefixes all log files with the given prefix |
`--num-neurons <size_t>` | `-n <size_t>` | Starts a simulation with approximately the given number of neurons | Excludes `--file` and `--num-neurons-per-rank` |
`--num-neurons-per-rank <size_t>` |  | Starts a simulation with the specified number of neurons per rank | Excludes `--file` and `--num-neurons` |
`--file <file-path>`| `-f <file-path>` | Starts a simulation with the neurons from the specified file | Excludes `--num-neurons` and `--num-neurons-per-rank`
`--graph <file-path>` | `-g <file-path>` | Starts a simulation with an initial connectivity from the specified file | Requires `--file`
`--enable-interrupts <file-path>` | | Specifies the enable interrupts for a simulation | 
`--disable-interrupts <file-path>` | | Specified the disable interrupts for a simulation |
`--creation-interrupts <file-path>` | | Specified the creation interrupts for a simulation |
`--algorithm <enum>` | `-a <enum>` | Chooses the algorithm that is used to find the target neurons, can be *naive*, *barnes-hut*, *barnes-hut-inverted*, or *fast-multipole-method*
`--theta <double>` | `-t <double>` | Sets the acceptance criterion for the Barnes-Hut-Algorithm | Must be from `(0, 0.5]`, requires `--algorithm barnes-hut` or `--algorithm barnes-hut-inverted`. Default is `0.3`
`--kernel-type <enum>`| | Chooses the probability kernel, can be *gamma*, *gaussian*, *linear*, or *weibull* | Default is `gaussian`
`--gamma-k <double>` | | Specifies the shape parameter for the gamma probability kernel | Default is `1.0`
`--gamma-theta <double>` | | Specifies the scale parameter for the gamma probability kernel | Default is `1.0`
`--gaussian-sigma <double>` | | Specifies the scaling parameter for the gaussian probability kernel | Default is `750.0`
`--gaussian-mu <double>` | | Specifies the offset parameter for the gaussian probability kernel | Default is `0.0`
`--linear-cutoff <double>` | | Specifies the cut-off parameter for the linear probability kernel | Default is $\infty$
`--weibull-k <double>` | | Specifies the shape parameter for the weibull probability kernel | Default is `1.0`
`--weibull-b <double>` | | Specifies the scale parameter for the weibull probability kernel | Default is `1.0`
`--neuron-model <enum>`| | Chooses the neuron model, can be *poisson*, *izhikevich*, *aeif*, or *fitzhughnagumo* | Default is `poisson`
`--base-background-activity <double>` | | Specifies the base background activity by which all neurons are excited | Must be non-negative. The background activity is calculated as *base + N(mean, stddev)*
`--base-background-mean <double>` | | Specifies the mean background activity by which all neurons are excited | The background activity is calculated as *base + N(mean, stddev)*
`--base-background-stddev <double>` | | Specifies the standard deviation of the background activity by which all neurons are excited | Must be non-negative. The background activity is calculated as *base + N(mean, stddev)*
`--synapse_conductance <double>`| | The activity that is transfered when a connected neuron spikes. | Default is `0.03`
`--calcium-decay <double>` | | The decay constant of the intercellular calcium. | Default is `10000`
`--target-ca <double>` | | Sets the target calcium for all neurons | Must be from [0, 100]. Default is `0.7`
`--initial-ca <double>`| | Sets the initial calcium for all neurons. | Default is `0.0`
`--beta <double>` | | Specifies the amount of calcium ions that are gathered whenever a neuron spikes |
`--retract-ratio <double>`| | The ratio by which vacant synapses retract. | Default is `0.0`
`--synaptic-elements-lower-bound <double>` | | Sets the lower bound of initial synaptic elements | Must be from [0, $\infty$)
`--synaptic-elements-upper-bound <double>` | | Sets the upper bound of initial synaptic elements | Must be from [0, $\infty$) and not less than the lower bound
`--growth-rate <double>` | | Specifies the growth rate of the synaptic elements | Must be from [0, 1]
`--min-calcium-axons <double>` | | Specified the minimum intercellular calcium for axons to grow. | Default is `0.4`
`--min-calcium-excitatory-dendrites <double>` | | Specified the minimum intercellular calcium for excitatory dendrites to grow. | Default is `0.1`
`--min-calcium-inhibitory-dendrites <double>` | | Specified the minimum intercellular calcium for inhibitory dendrites to grow. | Default is `0.0`


### In case you have compiled the program with MPI, you can also prefix the command with: `mpiexec -n 4 ...`
However, if you do so, the number of MPI processes must be a multiple of 2.
